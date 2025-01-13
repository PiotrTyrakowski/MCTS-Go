#include <cuda.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cassert>
#include <string>
#include <memory>

////////////////////////////////////////////////////////////////////////////////
// Basic constants and structures
////////////////////////////////////////////////////////////////////////////////

static const int N = 19;          // Board dimension
static const int NN = N * N;      // Number of points on the board

// Stone colors
static const unsigned char WHITE = 'O';
static const unsigned char BLACK = 'X';
static const unsigned char EMPTY = '.';
static const unsigned char DEAD  = '?';  // for marking territory or dead stones

// Helper to swap colors
__host__ __device__
unsigned char swap_colors(unsigned char color) {
    if (color == BLACK)  return WHITE;
    if (color == WHITE)  return BLACK;
    return color;
}

// Flatten and unflatten
__host__ __device__
int flatten(int r, int c) {
    return r * N + c;
}

__host__ __device__
void unflatten(int idx, int &r, int &c) {
    r = idx / N;
    c = idx % N;
}

__host__ __device__
bool on_board(int r, int c) {
    return (r >= 0 && r < N && c >= 0 && c < N);
}

////////////////////////////////////////////////////////////////////////////////
// Precompute neighbor arrays for each intersection
////////////////////////////////////////////////////////////////////////////////
__device__ __constant__ int d_neighbors[NN][4];
static int h_neighbors[NN][4];

__host__
void init_neighbors_cpu() {
    for (int fc = 0; fc < NN; fc++) {
        int r, c;
        unflatten(fc, r, c);
        int idx = 0;
        // up
        if (on_board(r-1, c)) {
            h_neighbors[fc][idx++] = flatten(r-1, c);
        }
        // down
        if (on_board(r+1, c)) {
            h_neighbors[fc][idx++] = flatten(r+1, c);
        }
        // left
        if (on_board(r, c-1)) {
            h_neighbors[fc][idx++] = flatten(r, c-1);
        }
        // right
        if (on_board(r, c+1)) {
            h_neighbors[fc][idx++] = flatten(r, c+1);
        }
        // Fill the rest with -1 to indicate no neighbor
        while (idx < 4) {
            h_neighbors[fc][idx++] = -1;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// find_reached (like in the Python reference) - CPU version
// This returns all stones connected to fc (the chain) and "reached" points
// that are not of that color (neighbors)
////////////////////////////////////////////////////////////////////////////////
void find_reached_cpu(const std::vector<unsigned char> &board,
                      int fc,
                      std::vector<int> &chain_out,
                      std::vector<int> &reached_out)
{
    chain_out.clear();
    reached_out.clear();

    unsigned char color = board[fc];

    std::vector<bool> chain_flag(NN, false);
    std::vector<bool> reached_flag(NN, false);

    std::vector<int> frontier;
    frontier.push_back(fc);
    chain_flag[fc] = true;

    while (!frontier.empty()) {
        int current_fc = frontier.back();
        frontier.pop_back();

        // Check neighbors
        for (int i = 0; i < 4; i++) {
            int fn = h_neighbors[current_fc][i];
            if (fn < 0) continue;  // invalid neighbor

            if (board[fn] == color) {
                if (!chain_flag[fn]) {
                    chain_flag[fn] = true;
                    frontier.push_back(fn);
                }
            } else {
                // different color or empty
                reached_flag[fn] = true;
            }
        }
    }

    // Gather results
    for (int i = 0; i < NN; i++) {
        if (chain_flag[i])   chain_out.push_back(i);
        if (reached_flag[i]) reached_out.push_back(i);
    }
}

////////////////////////////////////////////////////////////////////////////////
// maybe_capture_stones - CPU version
// If the chain connected to fc has no liberty, remove it (fill with EMPTY).
// Return the removed stones (chain).
////////////////////////////////////////////////////////////////////////////////
std::vector<int> maybe_capture_stones_cpu(std::vector<unsigned char> &board, int fc)
{
    std::vector<int> chain, reached;
    find_reached_cpu(board, fc, chain, reached);

    // Check if there's an EMPTY in reached
    bool has_liberty = false;
    for (auto fr : reached) {
        if (board[fr] == EMPTY) {
            has_liberty = true;
            break;
        }
    }

    if (!has_liberty) {
        // capture
        for (auto fstone : chain) {
            board[fstone] = EMPTY;
        }
        return chain;
    }
    return {};
}

////////////////////////////////////////////////////////////////////////////////
// play_move_incomplete (somewhat from Python code) - CPU
// This does not handle Ko fully. It just places a stone, captures adjacency.
////////////////////////////////////////////////////////////////////////////////
bool play_move_incomplete_cpu(std::vector<unsigned char> &board,
                              int fc,
                              unsigned char color)
{
    if (board[fc] != EMPTY) {
        // illegal
        return false;
    }
    board[fc] = color;
    
    unsigned char opp_color = swap_colors(color);

    // gather neighbors
    std::vector<int> my_stones;
    std::vector<int> opp_stones;
    for (int i = 0; i < 4; i++) {
        int fn = h_neighbors[fc][i];
        if (fn < 0) continue;
        if (board[fn] == color) {
            my_stones.push_back(fn);
        } else if (board[fn] == opp_color) {
            opp_stones.push_back(fn);
        }
    }

    // capture opponent
    for (auto fs : opp_stones) {
        maybe_capture_stones_cpu(board, fs);
    }
    // capture self if any group is suicide (like in normal rules)
    for (auto fs : my_stones) {
        maybe_capture_stones_cpu(board, fs);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Simple scoring function: count black stones - white stones
// This does not implement territory scoring for a real game, but it's a start.
////////////////////////////////////////////////////////////////////////////////
int score_cpu(const std::vector<unsigned char> &board)
{
    int black_count = 0;
    int white_count = 0;
    for (int i = 0; i < NN; i++) {
        if (board[i] == BLACK) {
            black_count++;
        } else if (board[i] == WHITE) {
            white_count++;
        }
    }
    return black_count - white_count;
}

////////////////////////////////////////////////////////////////////////////////
// Random Playout on GPU
//  - This kernel receives an array of boards, each board is a possible child
//    from a given node in the MCTS, and runs a random playout until some
//    "end" condition (like fill all intersections or no moves).
//  - The results are stored for each simulation so that the MCTS can
//    do a typical backprop.
////////////////////////////////////////////////////////////////////////////////

__global__
void random_playout_kernel(unsigned char *d_boards, // All boards in a 2D array [ num_boards x NN ]
                           int num_boards,
                           int *d_results,          // one result per board
                           curandState *states,
                           unsigned char to_move)   // whose turn it is
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_boards) return;

    // Each thread: run a random playout on board idx
    // We store board in d_boards[idx * NN : (idx+1) * NN-1]
    unsigned char local_board[NN];
    for (int i = 0; i < NN; i++) {
        local_board[i] = d_boards[idx * NN + i];
    }

    // set up RNG
    curandState localState = states[idx];

    // We do a simple random playout: pick random empty moves until full or no moves
    unsigned char current = to_move;

    // For practical GPU usage, you might limit the max number of moves to keep things short
    for (int step = 0; step < NN * 2; step++) {
        // gather empties
        int empties[NN];
        int empties_count = 0;
        for (int i = 0; i < NN; i++) {
            if (local_board[i] == EMPTY) {
                empties[empties_count++] = i;
            }
        }
        if (empties_count == 0) {
            // no moves => break
            break;
        }

        // pick random empty
        int choice = curand(&localState) % empties_count;
        int fc = empties[choice];
        
        // place stone
        // we do a naive approach here: no suicide check, or partial check 
        // (for real usage, you'd adapt `play_move_incomplete_cpu` or a GPU version).
        local_board[fc] = current;

        // switch color
        current = swap_colors(current);
    }

    // compute final score: black_count - white_count
    int black_count = 0, white_count = 0;
    for (int i = 0; i < NN; i++) {
        if (local_board[i] == BLACK)  black_count++;
        if (local_board[i] == WHITE)  white_count++;
    }
    d_results[idx] = black_count - white_count;

    // store RNG state back
    states[idx] = localState;
}

////////////////////////////////////////////////////////////////////////////////
// MCTS Node structure (CPU side)
////////////////////////////////////////////////////////////////////////////////
struct MCTSNode {
    std::vector<unsigned char> board;   // Current board state at this node
    unsigned char to_move;             // Whose turn is it in this node
    int visits;
    double wins;  // for the player to_move (or you can store separately)
    
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> valid_moves; // store the moves that create these children

    MCTSNode(const std::vector<unsigned char> &b, unsigned char tm)
        : board(b), to_move(tm), visits(0), wins(0.0)
    {}

    // Expand child nodes for each valid move
    void expand() {
        // gather empty points
        valid_moves.clear();
        for (int i = 0; i < NN; i++) {
            if (board[i] == EMPTY) {
                valid_moves.push_back(i);
            }
        }
        // create child boards
        for (auto mv : valid_moves) {
            std::vector<unsigned char> child_board = board;
            play_move_incomplete_cpu(child_board, mv, to_move);
            std::unique_ptr<MCTSNode> child(
                new MCTSNode(child_board, swap_colors(to_move))
            );
            children.push_back(std::move(child));
        }
    }

    bool is_leaf() const {
        return children.empty();
    }
};

////////////////////////////////////////////////////////////////////////////////
// MCTS: selection function (simplified UCB1)
////////////////////////////////////////////////////////////////////////////////
static inline double ucb_value(int parent_visits, double child_wins, int child_visits, double c=1.4) {
    if (child_visits == 0) {
        return 1e9; // effectively infinite
    }
    return (child_wins / child_visits) + c * sqrt(log((double)parent_visits) / child_visits);
}

MCTSNode* select_child_uct(MCTSNode* node) {
    MCTSNode *best = nullptr;
    double best_value = -1e30;
    for (size_t i = 0; i < node->children.size(); i++) {
        MCTSNode *c = node->children[i].get();
        double val = ucb_value(node->visits, c->wins, c->visits);
        if (val > best_value) {
            best_value = val;
            best = c;
        }
    }
    return best;
}

////////////////////////////////////////////////////////////////////////////////
// MCTS: random rollout (on CPU - naive)
////////////////////////////////////////////////////////////////////////////////
double cpu_random_rollout(const std::vector<unsigned char> &board, unsigned char to_move)
{
    // copy board
    std::vector<unsigned char> sim_board = board;
    unsigned char current = to_move;

    // Very naive random approach
    for (int step = 0; step < NN*2; step++) {
        std::vector<int> empties;
        empties.reserve(NN);
        for (int i = 0; i < NN; i++) {
            if (sim_board[i] == EMPTY) empties.push_back(i);
        }
        if (empties.empty()) {
            break;
        }
        int idx = rand() % empties.size();
        play_move_incomplete_cpu(sim_board, empties[idx], current);
        current = swap_colors(current);
    }

    int final_score = score_cpu(sim_board);
    // from the perspective of to_move:
    // if to_move == BLACK, then final_score > 0 => to_move wins
    // if to_move == WHITE, then final_score < 0 => to_move wins
    // we do a simple "win=1, lose=0" for MCTS
    if (to_move == BLACK) {
        return (final_score > 0) ? 1.0 : 0.0;
    } else {
        return (final_score < 0) ? 1.0 : 0.0;
    }
}

////////////////////////////////////////////////////////////////////////////////
// MCTS: the tree search itself
////////////////////////////////////////////////////////////////////////////////
double mcts_search(MCTSNode *root, int depth_limit=1000)
{
    // 1. Selection
    MCTSNode *node = root;
    int depth = 0;
    while (!node->is_leaf() && depth < depth_limit) {
        node = select_child_uct(node);
        depth++;
    }

    // 2. Expansion
    if (node->visits > 0) {
        // expand only if not visited yet
        node->expand();
        if (!node->children.empty()) {
            // pick one child at random for playout
            node = node->children[rand() % node->children.size()].get();
        }
    }

    // 3. Simulation
    double result = cpu_random_rollout(node->board, node->to_move);

    // 4. Backpropagation
    // climb up, flipping perspective each time
    MCTSNode *cur = node;
    unsigned char perspective = cur->to_move; // perspective for result
    while (cur) {
        cur->visits += 1;
        // If cur->to_move is the same color that originally rolled out,
        // we must interpret the result properly. 
        // An easy approach: if perspective == cur->to_move, add 'result';
        // else add '1.0 - result'. But below we keep it simple and assume
        // 'result' is from the viewpoint of cur->to_move. This can vary
        // depending on your design choice.
        if (cur->to_move == perspective) {
            cur->wins += result;
        } else {
            cur->wins += (1.0 - result);
        }

        // we don't have a parent pointer in MCTSNode, so in a real system
        // you'd store a pointer to the parent or implement a stack during
        // selection. For simplicity, we stop here.
        // In practice, you’d need a parent pointer or a recursion stack.
        break;
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
// Example usage
////////////////////////////////////////////////////////////////////////////////
int main()
{
    // Initialize neighbor info on CPU
    init_neighbors_cpu();

    // Copy neighbors to device constant memory
    cudaMemcpyToSymbol(d_neighbors, h_neighbors, sizeof(h_neighbors));

    // Create an empty board
    std::vector<unsigned char> board(NN, EMPTY);

    // Small test: place a few stones
    board[flatten(3,3)] = BLACK;
    board[flatten(3,4)] = WHITE;
    board[flatten(4,4)] = BLACK;

    // Create root node
    MCTSNode root(board, BLACK);

    // Expand the root to get possible moves
    root.expand();

    // Basic loop of MCTS
    int iterations = 1000;
    srand(1234); // CPU-based random seed
    for (int i = 0; i < iterations; i++) {
        mcts_search(&root);
    }

    // Choose best move from root
    double best_winrate = -1.0;
    int best_move = -1;
    for (size_t i = 0; i < root.children.size(); i++) {
        const MCTSNode *child = root.children[i].get();
        double wr = child->wins / (child->visits + 1e-9);
        if (wr > best_winrate) {
            best_winrate = wr;
            best_move = root.valid_moves[i];
        }
    }

    // Print the best move
    int r, c;
    unflatten(best_move, r, c);
    std::cout << "Best move is (" << r << ", " << c << ") with winrate "
              << best_winrate << "\n";

    // Demo: place it on the root board
    play_move_incomplete_cpu(root.board, best_move, BLACK);

    // Print final board
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            std::cout << (char)root.board[flatten(row,col)];
        }
        std::cout << std::endl;
    }

    return 0;
}
