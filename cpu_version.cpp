/****************************************************
 * A Simple C++ Go (19x19) Engine with MCTS
 * 
 * This sample aims to illustrate:
 *   1. Board representation
 *   2. Move legality (captures, suicide, ko)
 *   3. Monte Carlo Tree Search (selection, expansion,
 *      simulation, backpropagation)
 *   4. Basic final score with komi
 *
 * Note: This code is for demonstration/learning 
 *       purposes. Many optimizations and advanced 
 *       features are possible in a real Go engine.
 ****************************************************/

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>

////////////////////////////////////////////////////
// Global constants
////////////////////////////////////////////////////
static const int N = 9;            // Board dimension
static const int NN = N * N;        // Number of intersections
static const int EMPTY = 0;
static const int BLACK = 1;
static const int WHITE = 2;

// Typically, standard komi in 19x19 might be 6.5-7.5
static const double KOMI = 5.5;

// For MCTS
static const int MCTS_SIMULATIONS = 1000;  // You can adjust
static const double UCB_C = 1.41421;       // sqrt(2) for UCB1

////////////////////////////////////////////////////
// Board utilities
////////////////////////////////////////////////////

// Flatten 2D coordinate (row, col) into 1D index
inline int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten 1D index -> (row, col)
inline std::pair<int,int> unflatten(int idx) {
    return {idx / N, idx % N};
}

// Check if a (row, col) is on board
inline bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// Return list of neighbors in 1D index form
static std::vector<std::vector<int>> build_neighbors() {
    std::vector<std::vector<int>> neighbors(NN);
    for(int r = 0; r < N; r++){
        for(int c = 0; c < N; c++){
            int fc = flatten(r, c);
            // up, down, left, right
            if(is_on_board(r-1, c)) neighbors[fc].push_back(flatten(r-1, c));
            if(is_on_board(r+1, c)) neighbors[fc].push_back(flatten(r+1, c));
            if(is_on_board(r, c-1)) neighbors[fc].push_back(flatten(r, c-1));
            if(is_on_board(r, c+1)) neighbors[fc].push_back(flatten(r, c+1));
        }
    }
    return neighbors;
}

// Global table of neighbors for each intersection
static const std::vector<std::vector<int>> NEIGHBORS = build_neighbors();

// Swap colors
inline int swap_color(int color) {
    if(color == BLACK) return WHITE;
    if(color == WHITE) return BLACK;
    return color;
}

////////////////////////////////////////////////////
// Data structure representing the game state
////////////////////////////////////////////////////
struct Position {
    // Board array, 1D, each entry ∈ {EMPTY, BLACK, WHITE}
    std::array<int, NN> board;
    // Ko point (if any); -1 means no Ko
    int ko;
    // Next player to move
    int to_move;

    bool pass_happened;

    bool is_game_over;

    std::unordered_set<int> empty_spaces;

    Position() : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) {
        board.fill(EMPTY);
        for(int i = 0; i < NN; i++) {
            empty_spaces.insert(i);
        }
    }

    // Print board to stdout (for debugging)
    void print() const {
        for(int r=0; r<N; r++){
            for(int c=0; c<N; c++){
                int fc = flatten(r, c);
                if(board[fc] == EMPTY) std::cout << ".";
                else if(board[fc] == BLACK) std::cout << "X";
                else if(board[fc] == WHITE) std::cout << "O";
            }
            std::cout << "\n";
        }
        std::cout << "Ko: " << ko << ", to_move: " 
                  << (to_move == BLACK ? "BLACK" : "WHITE") << "\n";
    }


};

// Helper: Bulk place stones of "color" into "positions"
void bulk_remove_stones(Position &pos, const std::vector<int> &stones) {
    for(int fc : stones) {
        pos.board[fc] = EMPTY;
        pos.empty_spaces.insert(fc);
    }
}

// BFS/DFS to find chain and neighbors
std::pair<std::vector<int>, std::vector<int>> find_reached(
    const Position &pos, int start)
{
    int color = pos.board[start];
    std::vector<int> chain;
    chain.reserve(NN);  // overshoot
    std::vector<int> reached;
    reached.reserve(NN);

    std::vector<bool> visited(NN, false);
    visited[start] = true;
    std::queue<int> frontier;
    frontier.push(start);
    chain.push_back(start);

    while(!frontier.empty()){
        int current = frontier.front();
        frontier.pop();
        for(int nb : NEIGHBORS[current]){
            if(pos.board[nb] == color && !visited[nb]){
                visited[nb] = true;
                frontier.push(nb);
                chain.push_back(nb);
            }
            else if(pos.board[nb] != color) {
                reached.push_back(nb);
            }
        }
    }
    return {chain, reached};
}

// Attempt to capture a chain if it has no liberties
// Returns the set of captured stones (if any).
std::vector<int> maybe_capture_stones(Position &pos, int fc) {
    auto [chain, reached] = find_reached(pos, fc);
    // If no empty point in 'reached', remove chain
    bool has_liberty = false;
    for(int r : reached) {
        if(pos.board[r] == EMPTY) {
            has_liberty = true;
            break;
        }
    }
    if(!has_liberty){
        // Capture them
        bulk_remove_stones(pos, chain);
        return chain;
    }
    return {};
}

// Check if fc is "ko-ish": the move just captured exactly 1 stone
// and left a single surrounded point with no liberties.
bool is_koish(const Position &pos, int fc) {
    if(pos.board[fc] != EMPTY) return false;
    // If all neighbors are the same color (not empty), might be Ko
    int first_col = -1;
    for(int nb: NEIGHBORS[fc]) {
        if(pos.board[nb] != EMPTY) {
            if(first_col == -1) first_col = pos.board[nb];
            // If there's a mismatch or empty, not Ko
            if(pos.board[nb] != first_col) return false;
        } else {
            return false;
        }
    }
    return true; 
}

// Check if move at fc is legal (including ko, suicide)
bool is_legal_move(const Position &pos, int fc, int color) {
    if(fc < 0 || fc >= NN) return false;
    if(pos.board[fc] != EMPTY) return false; // must be empty
    if(fc == pos.ko) return false;           // can't retake Ko immediately

    // Make a copy to see if it results in suicide
    Position temp = pos;
    temp.board[fc] = color;
    // Capture opponent stones
    int opp_color = swap_color(color);

    // Need to see if we capture any neighbor groups of opposite color
    // or if the placed stone itself is captured (suicide).
    std::vector<int> neighborsOfMove = NEIGHBORS[fc];
    for(int nb : neighborsOfMove){
        if(temp.board[nb] == opp_color) {
            maybe_capture_stones(temp, nb);
        }
    }
    // Also check if we are suiciding
    auto captured = maybe_capture_stones(temp, fc);
    if(!captured.empty()) {
        // It's suicide if we just captured ourselves
        return false;
    }

    return true;
}

// Execute a move, returning a new Position
Position play_move(const Position &oldPos, int fc) {
    Position pos = oldPos; 
    int color = pos.to_move;

    if(fc == -1) {

        if(pos.pass_happened) {
            pos.is_game_over = true;
            return pos;
        }
        
        pos.pass_happened = true;
        pos.to_move = swap_color(color);
        return pos;
    }

    if(!is_legal_move(pos, fc, color)) {
        // In practice, you'd handle this more gracefully
        throw std::runtime_error("Illegal move attempted");
    }

    // Clear Ko
    pos.ko = -1;
    pos.board[fc] = color;
    pos.empty_spaces.erase(fc);  // Remove from empty spaces

    int opp_color = swap_color(color);

    // Capture any opponent stones adjacent
    int total_opp_captured = 0;
    for(int nb : NEIGHBORS[fc]){
        if(pos.board[nb] == opp_color) {
            auto captured = maybe_capture_stones(pos, nb);
            total_opp_captured += (int)captured.size();
        }
    }
    // Check for suicide (should never happen if is_legal_move passes)
    auto captured_self = maybe_capture_stones(pos, fc);
    if(!captured_self.empty()) {
        throw std::runtime_error("Suicide occurred unexpectedly");
    }

    // Check for Ko: if exactly 1 stone was captured and the new stone is in a 
    // one-point eye shape, set Ko
    if(total_opp_captured == 1 && is_koish(pos, fc)) {
        pos.ko = fc; 
    }

    // Next player
    pos.to_move = opp_color;

    return pos;
}

// For final scoring: 
//   We fill in connected empties by whichever color encloses them exclusively.
//   If both or neither, fill as neutral.
double final_score(const Position &pos) {
    // Copy board so we can fill in territory
    Position temp = pos;
    // We'll do a naive approach: for each empty region, see which color(s) border it.
    // If purely black => fill black, purely white => fill white, else fill neutral.
    for(int i = 0; i < NN; i++){
        if(temp.board[i] == EMPTY) {
            auto [chain, reached] = find_reached(temp, i);
            // Suppose the first neighbor color is the candidate
            int candidate = -1;
            bool mixed = false;
            for(int r: reached) {
                if(temp.board[r] == BLACK || temp.board[r] == WHITE) {
                    if(candidate < 0) candidate = temp.board[r];
                    else if(temp.board[r] != candidate) {
                        mixed = true;
                        break;
                    }
                }
            }
            if(!mixed && candidate > 0) {
                // fill chain with candidate
                for(int fc : chain) {
                    temp.board[fc] = candidate;
                }
            } else {
                // fill chain with '?' => treat as neutral
                for(int fc : chain) {
                    temp.board[fc] = -1;  // mark neutral
                }
            }
        }
    }
    // Now count
    int black_count = 0, white_count = 0;
    for(int i=0; i<NN; i++){
        if(temp.board[i] == BLACK) black_count++;
        else if(temp.board[i] == WHITE) white_count++;
    }
    // White gets komi
    double score = (double)black_count - (double)white_count + 0.0;
    score -= KOMI; 
    return score; 
}

////////////////////////////////////////////////////
// Monte Carlo Tree Search
////////////////////////////////////////////////////

static std::mt19937 rng((unsigned)std::time(nullptr));

// Simple Node for MCTS
struct Node {
    Position state;
    // Move that led to this node (fc). For root node, move_fc = -1
    int move_fc;
    // Statistics
    double wins;
    int visits;
    // Children
    std::vector<std::unique_ptr<Node>> children;
    // Parent
    Node *parent;
    // Valid moves from this position
    std::vector<int> legal_moves;

    Node(const Position &st, Node *p, int m)
        : state(st), move_fc(m), wins(0.0), visits(0), parent(p)
    {
        // Generate all possible moves
        for(int fc=0; fc<NN; fc++){
            if(is_legal_move(state, fc, state.to_move)) {
                legal_moves.push_back(fc);
            }
        }
    }
};

// UCB1 formula
double ucb(const Node &child, int total_visits) {
    if(child.visits == 0) {
        // Infinity in practice
        return 1e9;
    }
    double exploitation = child.wins / (double)child.visits;
    double exploration  = UCB_C * std::sqrt(std::log((double)total_visits) / (double)child.visits);
    return exploitation + exploration;
}

// Selection: descend the tree by picking child with max UCB
Node* select_child(Node *node) {
    Node* best_child = nullptr;
    double best_value = -1e9;
    for(auto &cptr : node->children) {
        double u = ucb(*cptr, node->visits);
        if(u > best_value) {
            best_value = u;
            best_child = cptr.get();
        }
    }
    return best_child;
}

// Expand one child (one untried move) 
Node* expand(Node *node) {
    // If no unexpanded moves, just return node
    if(node->children.size() == node->legal_moves.size()) {
        return node;
    }
    // Pick an unexpanded move
    int idx = (int)node->children.size();
    int move_fc = node->legal_moves[idx];
    // Create new child
    Position new_state = play_move(node->state, move_fc);
    node->children.push_back(std::make_unique<Node>(new_state, node, move_fc));
    return node->children.back().get();
}

// Simulation: do random playout until the game ends
// Return final score from the viewpoint of node->state.to_move
double simulate(Position st) {
    int first_player = st.to_move;
    int steps = 0;
    const int MAX_ROLLOUT_STEPS = NN * 5;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    while(steps < MAX_ROLLOUT_STEPS) {
        // Convert set to vector for random access
        std::vector<int> moves(st.empty_spaces.begin(), st.empty_spaces.end());
        
        // Add pass (-1) as a possible move only in late game
        if (st.empty_spaces.size() < NN/2) {
            moves.push_back(-1);
        }

        // Pick and try moves until we find a legal one
        int tries = 0;
        const int MAX_TRIES = 10;
        
        while (tries <= MAX_TRIES) {
            if (tries == MAX_TRIES) {
                st = play_move(st, -1);
                if (st.is_game_over) return final_score(st);
                break;
            }

            int idx = (int)(dist(rng) * moves.size());
            int move_fc = moves[idx];

          

            // Pass is always legal, other moves need checking
            if (move_fc == -1 || is_legal_move(st, move_fc, st.to_move)) {
                st = play_move(st, move_fc);
                if (st.is_game_over) 
                {
                    std::cout << "Game over after " << steps << " steps. Final score: " << final_score(st) << "\n";
                    return final_score(st);
                }
                break;
            }
            tries++;
        }

        
    }

    double sc = final_score(st);
    return (first_player == BLACK) ? (sc > 0.0 ? 1.0 : 0.0) 
                                  : (sc < 0.0 ? 1.0 : 0.0);
}

// Backpropagate result
void backprop(Node *node, double result) {
    while(node) {
        node->visits += 1;
        node->wins   += result;
        node = node->parent;
        // Flip result so that each parent sees from their perspective
        result = 1.0 - result; 
    }
}

// Perform one MCTS iteration
void mcts_iteration(Node *root) {
    // 1. Selection
    Node *node = root;
    while(!node->children.empty() && 
           (int)node->children.size() == (int)node->legal_moves.size()) {
        node = select_child(node);
    }
    // 2. Expansion
    node = expand(node);
    // 3. Simulation
    double result = simulate(node->state);
    // 4. Backprop
    backprop(node, result);
}

// Get best move from root after MCTS
int best_move(Node *root) {
    // We pick the move with the highest visitation count
    int best_fc = -1;
    double best_visits = -1.0;
    for(auto &cptr : root->children) {
        if(cptr->visits > best_visits) {
            best_visits = (double)cptr->visits;
            best_fc = cptr->move_fc;
        }
    }
    return best_fc;
}

////////////////////////////////////////////////////
// Example main
////////////////////////////////////////////////////
int main() {
    // Create initial position
    Position rootPos;
    std::unique_ptr<Node> root = std::make_unique<Node>(rootPos, nullptr, -1);

    for(int moveNumber = 1; moveNumber <= 201; moveNumber++) {
        std::cout << "Running MCTS iteration for move #" << moveNumber << "...\n"; // Debug output

        // Run MCTS
        for(int i=0; i<MCTS_SIMULATIONS * 2 ; i++){
            mcts_iteration(root.get());
        }

        std::cout << "MCTS iterations completed for move #" << moveNumber << "\n"; // Debug output

        // Pick best move
        int fc = best_move(root.get());
        if(fc < 0) {
            std::cout << "No moves available. Game ends.\n";
            break;
        }

        // Display chosen move
        auto rc = unflatten(fc);
        std::cout << "Move #" << moveNumber << " for "
                  << (root->state.to_move == BLACK ? "Black" : "White")
                  << " => (" << rc.first << "," << rc.second << ")\n";

        // Apply the move
        Position newPos = play_move(root->state, fc);
        // Print board
        newPos.print();

        // Create new root node
        root = std::make_unique<Node>(newPos, nullptr, -1);
    }

    // End: compute final score
    double score = final_score(root->state);
    std::cout << "Final Score (Black - White - Komi): " << score << "\n";
    if(score > 0) {
        std::cout << "BLACK wins by " << score << "!\n";
    } else {
        std::cout << "WHITE wins by " << -score << "!\n";
    }

    return 0;
}
