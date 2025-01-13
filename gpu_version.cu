/****************************************************
 * A Simple C++ Go (9x9) Engine with MCTS + CUDA 
 * (Simulation on GPU)
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

// Include CUDA headers
#include <cuda_runtime.h>
#include <curand_kernel.h>

////////////////////////////////////////////////////
// Global constants
////////////////////////////////////////////////////
static const int N = 9;             // Board dimension
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
// CUDA config
////////////////////////////////////////////////////
static const int THREADS_PER_BLOCK = 128;  // Example
static const int MAX_GPU_PLAYOUTS  = 1024; // Example max

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
// GPU Data Structures
////////////////////////////////////////////////////

/**
 * We define a simplified device-friendly struct to hold
 * the board information and current state needed for 
 * random playouts.
 */
struct GPUPosition {
    // Board in 1D, each entry in {EMPTY, BLACK, WHITE}
    int board[NN];
    
    // Ko point (-1 if none)
    int ko;
    
    // Next player to move
    int to_move;

    // 1 means game over, 0 means not over
    int is_game_over;
    
    // Track how many empty spots remain
    int empty_count;

    // Indicate that the last move was pass (to track double-pass -> game over).
    int pass_happened;
};

/**
 * We'll also flatten neighbors into a device array:
 * Each intersection i has up to 4 neighbors. If neighbor doesn't exist,
 * store -1 in that slot.
 */
__constant__ int d_neighbors[NN * 4]; 
// We'll fill this from the host code using a helper.

////////////////////////////////////////////////////
// Some device helper functions
////////////////////////////////////////////////////
__device__ inline void bulk_remove_stones_device(GPUPosition &pos, const int *chain, int chain_size)
{
    // Remove stones in the chain
    for(int i = 0; i < chain_size; i++) {
        int fc = chain[i];
        pos.board[fc] = EMPTY;
    }
}

//
// BFS on the device to find chain and reached
// chain[] will contain the stones of the group
// reached[] will contain distinct neighbors
// Return the size of chain and reached via out params
//
__device__ void find_reached_device(const GPUPosition &pos, int start,
                                    int *chain, int &chain_size,
                                    int *reached, int &reached_size)
{
    const int color = pos.board[start];
    bool visited[NN];
    for(int i=0; i<NN; i++) visited[i] = false;

    int queue_fc[NN];
    int queue_head = 0, queue_tail = 0;

    visited[start] = true;
    queue_fc[queue_tail++] = start;

    chain_size = 0;
    reached_size = 0;

    while(queue_head < queue_tail) {
        int current = queue_fc[queue_head++];
        chain[chain_size++] = current;

        // Check up to 4 neighbors
        int base = current * 4;
        for(int k=0; k<4; k++) {
            int nb = d_neighbors[base + k];
            if(nb < 0) continue;  // invalid neighbor
            int nb_color = pos.board[nb];
            if(nb_color == color && !visited[nb]) {
                visited[nb] = true;
                queue_fc[queue_tail++] = nb;
            }
            else if(nb_color != color) {
                // This neighbor is "reached"
                reached[reached_size++] = nb;
            }
        }
    }
}

// Attempt to capture a chain if it has no liberties
// Returns number of captured stones. The chain is removed from pos if captured.
__device__ int maybe_capture_stones_device(GPUPosition &pos, int fc)
{
    int chain[NN];
    int reached[NN];
    int chain_size = 0;
    int reached_size = 0;

    find_reached_device(pos, fc, chain, chain_size, reached, reached_size);

    // Check for any liberty
    bool has_liberty = false;
    for(int i=0; i<reached_size; i++){
        int r = reached[i];
        if(pos.board[r] == EMPTY) {
            has_liberty = true;
            break;
        }
    }

    if(!has_liberty) {
        // Capture
        bulk_remove_stones_device(pos, chain, chain_size);
        return chain_size;
    }
    return 0;
}

// Check if a move is legal
// This is a minimal check for pass or board emptiness or Ko
// plus the standard "no immediate suicide" check.
__device__ bool is_legal_move_device(const GPUPosition &pos, int fc, int color)
{
    if(fc < 0 || fc >= NN) return false;              // out of range
    if(pos.board[fc] != EMPTY) return false;          // must be empty
    if(fc == pos.ko) return false;                    // can't retake Ko

    // Copy position
    GPUPosition temp = pos;
    // Place the stone
    temp.board[fc] = color;

    // Capture adjacent enemy stones
    const int opp_color = swap_color_device(color);
    int base = fc * 4;
    for(int k=0; k<4; k++){
        int nb = d_neighbors[base + k];
        if(nb < 0) continue;
        if(temp.board[nb] == opp_color) {
            maybe_capture_stones_device(temp, nb);
        }
    }

    // Also check if we just suicided
    // i.e. see if the newly-placed stone got captured
    int captured_self = maybe_capture_stones_device(temp, fc);
    if(captured_self > 0) {
        // it's a suicide
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

// Actually play a move on pos and return the updated pos
// pass is indicated by fc == -1
__device__ void play_move_device(GPUPosition &pos, int fc)
{
    const int color = pos.to_move;

    // pass
    if(fc == -1) {
        if(pos.pass_happened) {
            // game over
            pos.is_game_over = 1;
            return;
        }
        pos.pass_happened = 1;
        // swap color
        pos.to_move = swap_color_device(color);
        return;
    }

    // Assume is_legal_move_device was checked prior
    // Clear Ko
    pos.ko = -1;

    // Place the stone
    pos.board[fc] = color;

    // Decrement empty count
    pos.empty_count--;

    // capture opponent stones
    const int opp_color = swap_color_device(color);
    int base = fc * 4;
    int total_opp_captured = 0;
    for(int k=0; k<4; k++){
        int nb = d_neighbors[base + k];
        if(nb < 0) continue;
        if(pos.board[nb] == opp_color) {
            total_opp_captured += maybe_capture_stones_device(pos, nb);
        }
    }
    // check suicide (should not happen if is_legal_move passes)
    // ...

    // check Ko: 
    // If exactly 1 stone was captured and the new stone is in a 1-point eye
    // we skip the full check here for brevity but you'd do it if needed.

    // Next player
    pos.to_move = opp_color;

    // Reset pass info
    pos.pass_happened = 0;
}

// Very naive territory scoring. 
// For demonstration, we do it on GPU in a similar BFS style.
__device__ double final_score_device(const GPUPosition &pos)
{
    // We'll do a copy of the board array in local memory
    // to fill territory. (Expensive in real code.)
    GPUPosition temp = pos;
    
    // Mark territory
    for(int i=0; i<NN; i++){
        if(temp.board[i] == EMPTY) {
            // BFS to see who encloses it
            int chain[NN], reached[NN];
            int chain_size=0, reached_size=0;
            find_reached_device(temp, i, chain, chain_size, reached, reached_size);
            int candidate = -1;
            bool mixed = false;
            for(int r=0; r<reached_size; r++){
                int col = temp.board[reached[r]];
                if(col == BLACK || col == WHITE) {
                    if(candidate < 0) candidate = col;
                    else if(col != candidate) {
                        mixed = true;
                        break;
                    }
                }
            }
            if(!mixed && candidate > 0) {
                // fill
                for(int c=0; c<chain_size; c++){
                    temp.board[chain[c]] = candidate;
                }
            } else {
                // mark neutral
                for(int c=0; c<chain_size; c++){
                    temp.board[chain[c]] = -1; // neutral
                }
            }
        }
    }
    // now count black vs white
    int black_count=0, white_count=0;
    for(int i=0; i<NN; i++){
        if(temp.board[i] == BLACK) black_count++;
        else if(temp.board[i] == WHITE) white_count++;
    }
    double score = (double)black_count - (double)white_count;
    // apply komi
    score -= KOMI;
    return score;
}

////////////////////////////////////////////////////////////
// GPU Random Playout Kernel
////////////////////////////////////////////////////////////

/**
 * Each thread in this kernel runs one random playout starting from the same 
 * initial position (provided in d_rootPositions[blockIdx.x]). 
 *
 * We store the final result from the perspective of 'root to_move' in d_results[globalThreadId].
 */
__global__ void simulate_kernel(GPUPosition *d_rootPositions,
                                double *d_results,
                                int  simulationsPerBlock,
                                unsigned long long seed)
{
    // global thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= simulationsPerBlock) return;

    // We use blockIdx.x to pick which root position to simulate if needed
    // For simplicity, let's assume we only have 1 root position per kernel launch.
    GPUPosition localPos = d_rootPositions[0];

    // Setup cuRAND
    curandState state;
    curand_init(seed + tid, 0, 0, &state);

    // The color who moves first
    int first_player = localPos.to_move;

    const int MAX_ROLLOUT_STEPS = NN * 5;
    int steps = 0;

    while(!localPos.is_game_over && steps < MAX_ROLLOUT_STEPS) {
        // Build a small list of empties
        // For demonstration, we just gather them each time
        // (inefficient on GPU in practice!)
        int empties[NN];
        int eCount = 0;
        for(int i=0; i<NN; i++){
            if(localPos.board[i] == EMPTY) {
                empties[eCount++] = i;
            }
        }
        // Optionally add pass
        bool allowPass = (eCount < (NN/2));

        // If no empties, must pass
        if(eCount == 0) {
            play_move_device(localPos, -1);
            break;
        }

        // Try random moves
        int tries = 0;
        const int MAX_TRIES = 10;
        bool movePlayed = false;

        while(tries < MAX_TRIES) {
            // pick random index
            int rIndex = (int)(curand_uniform(&state) * (allowPass ? (eCount+1) : eCount));
            
            int candidate = (rIndex == eCount) ? -1 : empties[rIndex];
            if(candidate == -1) {
                // pass is always legal
                play_move_device(localPos, -1);
                movePlayed = true;
                break;
            }
            else {
                if(is_legal_move_device(localPos, candidate, localPos.to_move)) {
                    play_move_device(localPos, candidate);
                    movePlayed = true;
                    break;
                }
            }
            tries++;
        }

        // if we didn't find a legal move after tries, just pass
        if(!movePlayed) {
            play_move_device(localPos, -1);
        }

        steps++;
    }

    // compute final score
    double sc = final_score_device(localPos);
    // from perspective of first_player
    double result = 0.0;
    if(first_player == BLACK) {
        result = (sc > 0.0 ? 1.0 : 0.0);
    } else {
        result = (sc < 0.0 ? 1.0 : 0.0);
    }

    // store
    d_results[tid] = result;
}

////////////////////////////////////////////////////////////
// Host function to run GPU simulations
////////////////////////////////////////////////////////////
double simulate_on_gpu(const GPUPosition &rootPos, int num_simulations)
{
    // We limit the total # of simulations to MAX_GPU_PLAYOUTS for demonstration
    int sim_count = (num_simulations > MAX_GPU_PLAYOUTS) ? MAX_GPU_PLAYOUTS : num_simulations;

    // Allocate device arrays
    GPUPosition *d_rootPositions;
    double *d_results;

    cudaMalloc(&d_rootPositions, sizeof(GPUPosition));
    cudaMalloc(&d_results, sim_count * sizeof(double));

    // Copy rootPos to device
    cudaMemcpy(d_rootPositions, &rootPos, sizeof(GPUPosition), cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (sim_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    unsigned long long seed = (unsigned long long)time(nullptr);

    simulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_rootPositions,
                                                   d_results,
                                                   sim_count,
                                                   seed);
    cudaDeviceSynchronize();

    // Copy results back
    std::vector<double> results(sim_count);
    cudaMemcpy(results.data(), d_results, sim_count * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_rootPositions);
    cudaFree(d_results);

    // Simple: average the results
    double sum = 0.0;
    for(double r : results) {
        sum += r;
    }
    double avg = sum / (double)sim_count;
    return avg;
}

////////////////////////////////////////////////////////////
// Original CPU-based data structures for MCTS
////////////////////////////////////////////////////////////

struct Position {
    std::array<int, NN> board;
    int ko;
    int to_move;
    bool pass_happened;
    bool is_game_over;
    std::unordered_set<int> empty_spaces;

    Position() : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false)
    {
        board.fill(EMPTY);
        for(int i = 0; i < NN; i++) {
            empty_spaces.insert(i);
        }
    }

    void print() const {
        for(int r=0; r<N; r++){
            for(int c=0; c<N; c++){
                int fc = flatten_device(r, c);
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

// We still keep the CPU-based BFS, capturing, etc. for tree expansion, etc.
// ... (omitting details to keep this demonstration shorter)

////////////////////////////////////////////////////////////
// Convert CPU Position -> GPUPosition
////////////////////////////////////////////////////////////
GPUPosition to_GPUPosition(const Position &pos)
{
    GPUPosition gp;
    for(int i=0; i<NN; i++){
        gp.board[i] = pos.board[i];
    }
    gp.ko           = pos.ko;
    gp.to_move      = pos.to_move;
    gp.is_game_over = pos.is_game_over ? 1 : 0;
    gp.pass_happened = pos.pass_happened ? 1 : 0;

    gp.empty_count  = (int)pos.empty_spaces.size();
    return gp;
}

////////////////////////////////////////////////////////////
// Build neighbors on the host and copy to d_neighbors
////////////////////////////////////////////////////////////
static std::vector<int> build_neighbors_flat()
{
    // Each intersection i has up to 4 neighbors
    // We'll flatten into a single array d_neighbors[i*4 + {0..3}]
    // If neighbor is invalid, store -1
    std::vector<int> result(NN * 4, -1);
    for(int r=0; r<N; r++){
        for(int c=0; c<N; c++){
            int fc = flatten_device(r,c);
            int base = fc*4;
            int idx = 0;
            // up
            if(r-1 >= 0) result[base + idx++] = flatten_device(r-1,c);
            // down
            if(r+1 <  N) result[base + idx++] = flatten_device(r+1,c);
            // left
            if(c-1 >= 0) result[base + idx++] = flatten_device(r,c-1);
            // right
            if(c+1 <  N) result[base + idx++] = flatten_device(r,c+1);
        }
    }
    return result;
}


////////////////////////////////////////////////////////////
// CPU MCTS Node
////////////////////////////////////////////////////////////

struct Node {
    Position state;
    int move_fc;
    double wins;
    int visits;
    std::vector<std::unique_ptr<Node>> children;
    Node *parent;
    std::vector<int> legal_moves;

    Node(const Position &st, Node *p, int m)
        : state(st), move_fc(m), wins(0.0), visits(0), parent(p)
    {
        // (Compute legal_moves on CPU as before)
        // ...
    }
};

////////////////////////////////////////////////////////////
// The rest of MCTS: UCB, select, expand, etc. remain on CPU
// We'll just replace "simulate(...)" with a call to GPU.
////////////////////////////////////////////////////////////

// double cpu_simulate(Position st) {
//     int first_player = st.to_move;
//     int steps = 0;
//     const int MAX_ROLLOUT_STEPS = NN * 5;
    
//     std::uniform_real_distribution<double> dist(0.0, 1.0);

//     while(steps < MAX_ROLLOUT_STEPS) {
//         // Convert set to vector for random access
//         std::vector<int> moves(st.empty_spaces.begin(), st.empty_spaces.end());
        
//         // Add pass (-1) as a possible move only in late game
//         if (st.empty_spaces.size() < NN/2) {
//             moves.push_back(-1);
//         }

//         // Pick and try moves until we find a legal one
//         int tries = 0;
//         const int MAX_TRIES = 10;
        
//         while (tries <= MAX_TRIES) {
//             if (tries == MAX_TRIES) {
//                 st = play_move(st, -1);
//                 if (st.is_game_over) return final_score(st);
//                 break;
//             }

//             int idx = (int)(dist(rng) * moves.size());
//             int move_fc = moves[idx];

          

//             // Pass is always legal, other moves need checking
//             if (move_fc == -1 || is_legal_move(st, move_fc, st.to_move)) {
//                 st = play_move(st, move_fc);
//                 if (st.is_game_over) return final_score(st);
//                 break;
//             }
//             tries++;
//         }

        
//     }

//     double sc = final_score(st);
//     return (first_player == BLACK) ? (sc > 0.0 ? 1.0 : 0.0) 
//                                   : (sc < 0.0 ? 1.0 : 0.0);
// }

// We do an alternative that calls GPU
double simulate(const Position &st)
{
    // Convert st to GPUPosition
    GPUPosition gp = to_GPUPosition(st);

    // Suppose we want, for demonstration, e.g. 16 random playouts on GPU, 
    // and we’ll take the average result. 
    // In real MCTS, you might do just 1 or some smaller number.
    int num_sim = 16;
    double result = simulate_on_gpu(gp, num_sim);
    // Return the average => either 0..1
    return result;
}

// UCB, select_child, expand, backprop remain the same on CPU
// except we call the new "simulate()" (GPU version).

double ucb(const Node &child, int total_visits) {
    if(child.visits == 0) {
        return 1e9;
    }
    double exploitation = child.wins / (double)child.visits;
    double exploration  = UCB_C * std::sqrt(std::log((double)total_visits) / (double)child.visits);
    return exploitation + exploration;
}

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

Node* expand(Node *node) {
    // If all possible moves have been expanded, do nothing
    if (node->children.size() == node->legal_moves.size()) {
        return node;
    }

    // Pick the next unexpanded move
    int idx = static_cast<int>(node->children.size());
    int move_fc = node->legal_moves[idx];

    // Play this move to get a new state
    Position new_state = play_move(node->state, move_fc);

    // Create a new child node
    node->children.push_back(std::make_unique<Node>(new_state, node, move_fc));

    // Return a pointer to the newly-created child
    return node->children.back().get();
}

void backprop(Node *node, double result) {
    while(node) {
        node->visits += 1;
        node->wins   += result;
        node = node->parent;
        result = 1.0 - result;
    }
}

void mcts_iteration(Node *root) {
    Node *node = root;
    // 1. Selection
    while(!node->children.empty() && 
           (int)node->children.size() == (int)node->legal_moves.size()) 
    {
        node = select_child(node);
    }
    // 2. Expansion
    node = expand(node);
    // 3. Simulation (GPU-based now)
    double result = simulate(node->state);
    // 4. Backprop
    backprop(node, result);
}

int best_move(Node *root) {
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

////////////////////////////////////////////////////////////
// Main
////////////////////////////////////////////////////////////
int main()
{
    // ------------------------------------------------------
    // 1) Build and copy the neighbor table to the GPU
    //    (Assuming build_neighbors_flat() is already defined)
    // ------------------------------------------------------
    std::vector<int> host_neighbors = build_neighbors_flat();
    cudaMemcpyToSymbol(d_neighbors, host_neighbors.data(),
                       NN * 4 * sizeof(int), 0, cudaMemcpyHostToDevice);

    // ------------------------------------------------------
    // 2) Create the initial position and the root MCTS node
    // ------------------------------------------------------
    Position rootPos;  // an empty 9x9 board, BLACK to move first by default
    std::unique_ptr<Node> root = std::make_unique<Node>(rootPos, nullptr, -1);

    // ------------------------------------------------------
    // 3) MCTS main loop
    //    (We arbitrarily limit to 201 moves as an example.)
    // ------------------------------------------------------
    for (int moveNumber = 1; moveNumber <= 201; moveNumber++) {
        std::cout << "Running MCTS for move #" << moveNumber << "...\n";

        // Perform multiple MCTS iterations
        for (int i = 0; i < MCTS_SIMULATIONS; i++) {
            mcts_iteration(root.get());
        }

        // After MCTS, pick the best move from the root
        int fc = best_move(root.get());
        if (fc < 0) {
            std::cout << "No moves available. Game ends.\n";
            break;
        }

        // --------------------------------------------------
        // 4) Apply the chosen move on the CPU
        //    (Use your existing play_move or an equivalent.)
        // --------------------------------------------------
        Position newPos = play_move(root->state, fc);

        // Debug: print the move and the updated board
        auto rc = unflatten(fc);
        std::cout << "Move #" << moveNumber << " for "
                  << (root->state.to_move == BLACK ? "Black" : "White")
                  << " => (" << rc.first << "," << rc.second << ")\n";
        newPos.print();

        // Create a new root node from this updated state
        root = std::make_unique<Node>(newPos, nullptr, -1);

        // If the game is over, stop
        if (newPos.is_game_over) {
            std::cout << "Game over by double-pass or other condition.\n";
            break;
        }
    }

    // ------------------------------------------------------
    // 5) Final scoring (CPU-based)
    // ------------------------------------------------------
    double score = final_score(root->state);
    std::cout << "Final Score (Black - White - Komi): " << score << "\n";
    if (score > 0) {
        std::cout << "BLACK wins by " << score << "!\n";
    } else {
        std::cout << "WHITE wins by " << -score << "!\n";
    }

    // Done
    return 0;
}