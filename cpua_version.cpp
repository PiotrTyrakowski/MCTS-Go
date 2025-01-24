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
static const int N = 5;            // Board dimension
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

// BOARD_SIZE > row >= 0, BOARD_SIZE > col >= 0

struct BoardConfig {
    int board_size;
    double komi;

    constexpr BoardConfig(int board_size, double komi) : board_size(board_size), komi(komi) {}
};

inline int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten 1D index -> (row, col)
inline std::pair<int,int> unflatten(int idx) {
    return {idx / N , idx % N };
}

// Check if a (row, col) is on board
inline bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// Return list of neighbors in 1D index form`  
static std::vector<std::vector<int>> build_neighbors() {
    std::vector<std::vector<int>> neighbors(N * N);
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
template<int BOARD_SIZE>
struct Position {
    // Board array, 1D, each entry ∈ {EMPTY, BLACK, WHITE}
    std::array<int, BOARD_SIZE * BOARD_SIZE> board;
    // Ko point (if any); -1 means no Ko
    int ko;
    // Next player to move
    int to_move;
    
    bool pass_happened;

    bool is_game_over;

    std::unordered_set<int> empty_spaces;

    Position() : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) {
        board.fill(EMPTY);
        for(int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++) {
            empty_spaces.insert(i);
        }
    }

    // Print board to stdout (for debugging)
    void print() const {
        for(int r=0; r<BOARD_SIZE; r++){
            for(int c=0; c<BOARD_SIZE; c++){
                int fc = flatten(r, c);
                if(board[fc] == EMPTY) std::cout << ".";
                else if(board[fc] == BLACK) std::cout << "X";
                else if(board[fc] == WHITE) std::cout << "O";
            }
            std::cout << "\n";
        }
        std::cout << "Ko: " << ko << ", to_move: " << (to_move == BLACK ? "BLACK" : "WHITE") << ", pass_happened: " << pass_happened << ", is_game_over: " << is_game_over << "\n";
    }


};
 
// Helper: Bulk remove stones from board
template<int BOARD_SIZE>
void bulk_remove_stones(Position<BOARD_SIZE> &pos, const std::vector<int> &stones) {
    for(int fc : stones) {
        pos.board[fc] = EMPTY;
        pos.empty_spaces.insert(fc);
    }
}

// Finds two things:
// 1. A chain: all connected stones of the same color starting from 'start'
// 2. All points adjacent to this chain (reached points)
// Returns: pair of vectors {chain, reached}
template<int BOARD_SIZE>
std::pair<std::vector<int>, std::vector<int>> find_reached(
    const Position<BOARD_SIZE> &pos, int start)
{
    // Get the color of the starting point (BLACK, WHITE, or EMPTY)
    int color = pos.board[start];
    
    // Will store all connected stones of the same color
    std::vector<int> chain;
    chain.reserve(BOARD_SIZE * BOARD_SIZE);  // Pre-allocate max possible size
    
    // Will store all points adjacent to the chain
    std::vector<int> reached;
    reached.reserve(BOARD_SIZE * BOARD_SIZE);

    // Keep track of which points we've already processed
    std::vector<bool> visited(BOARD_SIZE * BOARD_SIZE, false);
    visited[start] = true;
    
    // BFS queue for exploring connected stones
    std::queue<int> frontier;
    frontier.push(start);
    chain.push_back(start);

    // Breadth-first search through connected stones
    while(!frontier.empty()){
        int current = frontier.front();
        frontier.pop();
        
        // Check all neighboring points
        for(int nb : NEIGHBORS[current]){
            if(pos.board[nb] == color && !visited[nb]){
                // Same color and not visited: add to chain
                visited[nb] = true;
                frontier.push(nb);
                chain.push_back(nb);
            }
            else if(pos.board[nb] != color) {
                // Different color or empty: add to reached points
                reached.push_back(nb);
            }
        }
    }
    
    return {chain, reached};
}

// Attempt to capture a chain if it has no liberties
// Returns the set of captured stones (if any).
template<int BOARD_SIZE>
std::vector<int> maybe_capture_stones(Position<BOARD_SIZE> &pos, int fc) {
    // Find all connected stones of the same color and their adjacent points
    auto [chain, reached] = find_reached<BOARD_SIZE>(pos, fc);

    // Check if the chain has any liberties (empty adjacent points)
    bool has_liberty = false;
    for(int r : reached) {
        if(pos.board[r] == EMPTY) {
            has_liberty = true;
            break;
        }
    }

    // If no liberties found, capture the entire chain
    if(!has_liberty){
        bulk_remove_stones<BOARD_SIZE>(pos, chain);
        return chain;  // Return the captured stones
    }
    return {};  // Return empty vector if no stones were captured
}

// Check if fc is "ko-ish": the move just captured exactly 1 stone
// and left a single surrounded point with no liberties.
template<int BOARD_SIZE>
bool is_koish_for_next_player(const Position<BOARD_SIZE> &pos, int maybe_ko_checker, int played_stone) {

    // Check if the played stone is empty
    if(pos.board[played_stone] == EMPTY) return false;

    // Check if the maybe_ko_checker is empty
    if(pos.board[maybe_ko_checker] != EMPTY) return false;


    int played_stone_color = pos.board[played_stone];

   
    for(int nb: NEIGHBORS[played_stone]) {

        if(pos.board[nb] == played_stone_color) {
            return false;
        }

        if(pos.board[nb] == EMPTY && nb != maybe_ko_checker) {
            return false;
        }

    }
    return true; 
}

// Check if move at fc is legal (including ko, suicide)
template<int BOARD_SIZE>
bool is_legal_move(const Position<BOARD_SIZE> &pos, int fc, int color) {
    if(fc < 0 || fc > BOARD_SIZE * BOARD_SIZE) return false;
    if(pos.board[fc] != EMPTY) return false; // must be empty
    if(fc == pos.ko) return false;           // can't retake Ko immediately
    if(fc == BOARD_SIZE * BOARD_SIZE) return true; // pass always legal
  
    // Make a copy to see if it results in suicide
    Position<BOARD_SIZE> temp = pos;
    temp.board[fc] = color;
    // Capture opponent stones
    int opp_color = swap_color(color);

    // Need to see if we capture any neighbor groups of opposite color
    // or if the placed stone itself is captured (suicide).
    std::vector<int> neighborsOfMove = NEIGHBORS[fc];
    for(int nb : neighborsOfMove){
        if(temp.board[nb] == opp_color) {
            maybe_capture_stones<BOARD_SIZE>(temp, nb);
        }
    }
    // Also check if we are suiciding
    auto captured = maybe_capture_stones<BOARD_SIZE>(temp, fc);
    if(!captured.empty()) {
        // It's suicide if we just captured ourselves
        return false;
    }

    return true;
}

// Execute a move, returning a new Position
// WE know that move is legal, so we don't need to check for suicide
template<int BOARD_SIZE>
Position<BOARD_SIZE> play_move(const Position<BOARD_SIZE> &oldPos, int fc) {
    Position<BOARD_SIZE> pos = oldPos; 
    int color = pos.to_move;

    // Pass move
    if(fc == BOARD_SIZE) {

        if(pos.pass_happened) {
            pos.is_game_over = true;
            return pos;
        }
        
        pos.pass_happened = true;
        pos.to_move = swap_color(color);
        return pos;
    }

  

    // Clear Ko
    pos.ko = -1;
    pos.board[fc] = color;
    pos.empty_spaces.erase(fc);  // Remove from empty spaces
    pos.pass_happened = false;

    int opp_color = swap_color(color);

    int maybe_ko_checker = -1;
    // Capture any opponent stones adjacent
    int total_opp_captured = 0;
    for(int nb : NEIGHBORS[fc]){
        if(pos.board[nb] == opp_color) {
            auto captured = maybe_capture_stones(pos, nb);
            total_opp_captured += (int)captured.size();
            if(total_opp_captured == 1){
                maybe_ko_checker = nb;
            }
        }
    }
   

    // Check for Ko: if exactly 1 stone was captured and the new stone is in a 
    // one-point eye shape, set Ko
    if(total_opp_captured == 1 && is_koish_for_next_player<BOARD_SIZE>(pos, maybe_ko_checker, fc)) {
        pos.ko = maybe_ko_checker; 
    }

    // Next player
    pos.to_move = opp_color;

    return pos;
}

// For final scoring: 
//   We fill in connected empties by whichever color encloses them exclusively.
//   If both or neither, fill as neutral.
template<int BOARD_SIZE>
double final_score(const Position<BOARD_SIZE> &pos) {
    // Copy board so we can fill in territory
    Position<BOARD_SIZE> temp = pos;
    // We'll do a naive approach: for each empty region, see which color(s) border it.
    // If purely black => fill black, purely white => fill white, else fill neutral.
    for(int i = 0; i < BOARD_SIZE * BOARD_SIZE; i++){
        if(temp.board[i] == EMPTY) {
            auto [chain, reached] = find_reached<BOARD_SIZE>(temp, i);
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
    for(int i=0; i<BOARD_SIZE * BOARD_SIZE; i++){
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
template<int BOARD_SIZE>
struct Node {
    // Parent
    Node<BOARD_SIZE> *parent;


    Position<BOARD_SIZE> state;
    // Move that led to this node (fc). For root node, move_fc = -1
    int move_fc;
    int move_number;
    int color_of_move;
    // Statistics

    double wins;
    int visits;
    double ucb_value;

    // Children
    std::vector<std::unique_ptr<Node>> children;
    std::vector<int> legal_moves;
    std::vector<double> children_ucb_values;
    std::vector<int> children_wins;
    std::vector<int> children_visits;


    Node(const Position &st, Node *p, int m, int move_number, int color_of_move, int wins, int visits, double ucb_value)
        : state(st), move_fc(m), move_number(move_number), color_of_move(color_of_move), wins(wins), visits(visits), ucb_value(ucb_value), parent(p)
    {

        // pass move
        legal_moves.push_back(BOARD_SIZE * BOARD_SIZE); // pass move
        children_wins.push_back(0);
        children_visits.push_back(0);
        children_ucb_values.push_back(0.0);



        // Generate all possible moves
        for(int fc=0; fc<BOARD_SIZE * BOARD_SIZE; fc++){
            if(is_legal_move(state, fc, state.to_move)) {
                legal_moves.push_back(fc);
                children_wins.push_back(0);
                children_visits.push_back(0);
                children_ucb_values.push_back(0.0);
            }
        }
    }
};

// UCB1 formula
template<int BOARD_SIZE>
double ucb_for_child(const Node<BOARD_SIZE> &child, int total_visits) {
    if(child.visits == 0) {
        // Infinity in practice
        return 1e9;
    }
    double exploitation = child.wins / (double)child.visits;
    double exploration  = UCB_C * std::sqrt(std::log((double)total_visits) / (double)child.visits);
    return exploitation + exploration;
}
    
// Selection: descend the tree by picking child with max UCB
template<int BOARD_SIZE>
Node<BOARD_SIZE>* select_child(Node<BOARD_SIZE> *node) {
    Node<BOARD_SIZE>* best_child = nullptr;
    double best_value = -1e9;
    for(auto &cptr : node->children) {
        double u = ucb<BOARD_SIZE>(*cptr, node->visits);
        if(u > best_value) {
            best_value = u;
            best_child = cptr.get();
        }
    }
    return best_child;
}

// Expand one child (one untried move) 
template<int BOARD_SIZE>
Node<BOARD_SIZE>* expand(Node<BOARD_SIZE> *node) {
    // If no unexpanded moves, just return node
    if(node->children.size() == node->legal_moves.size()) {
        return node;
    }
    // Pick an unexpanded move
    int idx = (int)node->children.size();
    int move_fc = node->legal_moves[idx];
    // Create new child
    Position<BOARD_SIZE> new_state = play_move<BOARD_SIZE>(node->state, move_fc);
    node->children.push_back(std::make_unique<Node<BOARD_SIZE>>(new_state, node, move_fc));
    return node->children.back().get();
}

template<int NUM_SIMULATIONS = 1>
int simulate(Position st) {
    // Which color began this simulation?
    int start_player = st.to_move;
    
    // Count how many times "start_player" ended up winning
    int wins_for_starter = 0;
    
    for (int sim = 0; sim < NUM_SIMULATIONS; sim++) {
        // We'll do a fresh copy for each playout
        Position current_st = st;
        const int MAX_ROLLOUT_STEPS = NN * 2;

        // Basic random generator
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        int steps = 0;
        while (!current_st.is_game_over && steps < MAX_ROLLOUT_STEPS) {
            // Build a naive list of candidate moves
            std::vector<int> moves(current_st.empty_spaces.begin(), current_st.empty_spaces.end());
            
            // Optionally allow a pass if the board is relatively full
            if (current_st.empty_spaces.size() < NN / 2) {
                moves.push_back(-1);  // pass
            }

            if(current_st.pass_happened){
                // 50% chance to pass
                if(dist(rng) < 0.5){
                    current_st = play_move(current_st, -1);
                    break;
                }
            }

            // Try up to 10 random picks from 'moves' to find a legal one
            bool move_found = false;
            for (int tries = 0; tries < 10; tries++) {
                int idx = static_cast<int>(dist(rng) * moves.size());
                int move_fc = moves[idx];

                // Check legality or pass
                if (move_fc == -1 || is_legal_move(current_st, move_fc, current_st.to_move)) {
                    current_st = play_move(current_st, move_fc);
                    move_found = true;
                    break;
                }
            }

            // If we still couldn't find anything, just pass:
            if(!move_found) {
                current_st = play_move(current_st, -1);
            }

            steps++;
        }

        // Game might have ended or we hit the rollout limit
        double sc = final_score(current_st); // Black - White - Komi
        // If sc > 0 => Black is ahead; if sc < 0 => White is ahead
        bool black_is_winner = (sc > 0.0);
        bool white_is_winner = (sc < 0.0);

        // Convert the final board outcome to "start_player" perspective:
        // if start_player == BLACK and black_is_winner => 1.0
        // if start_player == WHITE and white_is_winner => 1.0
        // else => 0.0
        if (start_player == BLACK && black_is_winner) {
            wins_for_starter++;
        }
        else if (start_player == WHITE && white_is_winner) {
            wins_for_starter++;
        }
        // If we want to handle draws differently, we can do so here

        return wins_for_starter;
    }
    return wins_for_starter;
}

template<int NUM_SIMULATIONS = 1>
void backprop(Node *node, int result) {
    while(node) {
        node->visits += NUM_SIMULATIONS;
        node->wins   += result;
        node = node->parent;
        // Flip result so that each parent sees from their perspective
        result = NUM_SIMULATIONS - result; 
    }
}

// Update the mcts_iteration function to use the templates
template<int NUM_SIMULATIONS = 1>
void mcts_iteration(Node *root) {
    // 1. Selection
    Node *node = root;
    while(!node->children.empty() && 
           (int)node->children.size() == (int)node->legal_moves.size()) {
        node = select_child(node);
    }
    // 2. Expansion
    node = expand(node);
    // 3. Simulation with multiple games
    double result = simulate<NUM_SIMULATIONS>(node->state);  // Example: run 5 simulations
    // 4. Backprop with corresponding number of simulations
    backprop<NUM_SIMULATIONS>(node, result);
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
        for(int i=0; i<MCTS_SIMULATIONS * 10; i++){
            mcts_iteration<10000>(root.get());
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
