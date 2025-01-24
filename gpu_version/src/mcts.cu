#include "mcts.cuh"
#include "constants.cuh"
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
#include <unordered_set>
#include <vector>

static std::mt19937 rng((unsigned)std::time(nullptr));

Node::Node(const Position &st, Node *p, int m, int move_num, int color_of_move)
    : parent(p), state(st), move_fc(m), move_number(move_num), color_of_move(color_of_move),
      wins(0.0), visits(0), ucb_value(0.0)
{
    for(int fc = 0; fc < NN; fc++){
        if(is_legal_move(state, fc, state.to_move)) {
            legal_moves.push_back(fc);
        }
    }
    legal_moves.push_back(NN);
}

double ucb_for_child(const Node &child, int total_visits) {
    if(child.visits == 0) {
        return 1e9; 
    }
    double exploitation = child.wins / (double)child.visits;
    double exploration  = UCB_C * std::sqrt(std::log((double)total_visits) / (double)child.visits);
    return exploitation + exploration;
}

Node* select_child(Node *node) {
    Node* best_child = nullptr;
    double best_value = -1e10;
    for(auto &cptr : node->children) {
        double val = ucb_for_child(*cptr, node->visits);
        if(val > best_value) {
            best_value = val;
            best_child = cptr.get();
        }
    }
    return best_child;
}

void expand(Node *node) {
    if(node->state.is_game_over) {
        return;
    }
    int next_player = node->state.to_move;
    for(auto move_fc : node->legal_moves) {
        Position new_state = play_move(node->state, move_fc);
        node->children.push_back(std::make_unique<Node>(new_state, node, move_fc,
                                                        node->move_number+1, next_player));
    }
}


void simulate_node(Node *nodes, int n_simulations_per_child) {
    int child_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (child_idx >= num_children) return;

    Node child = nodes[child_indices[child_idx]];
    Position *dev_positions;
    int *dev_results;

    cudaMalloc(&dev_positions, n_simulations_per_child * sizeof(Position));
    cudaMalloc(&dev_results, n_simulations_per_child * sizeof(int));


    Position *host_positions = new Position[n_simulations_per_child];
    for(int i = 0; i < n_simulations_per_child; i++) {
        host_positions[i] = child.state;
    }
    cudaMemcpy(dev_positions, host_positions, n_simulations_per_child * sizeof(Position), cudaMemcpyHostToDevice);
    delete[] host_positions;

    int threads_per_block = 256;
    int blocks = (n_simulations_per_child + threads_per_block - 1) / threads_per_block;
    simulate_position<<<blocks, threads_per_block>>>(dev_positions, dev_results, n_simulations_per_child, NN, seed_offset + child_idx);

    cudaDeviceSynchronize();

    int *host_results = new int[n_simulations_per_child];
    cudaMemcpy(host_results, dev_results, n_simulations_per_child * sizeof(int), cudaMemcpyDeviceToHost);

    int total_wins = 0;
    for(int i = 0; i < n_simulations_per_child; i++) {
        total_wins += host_results[i];
    }

    atomicAdd(&(nodes[child_indices[child_idx]].wins), (double)total_wins);
    atomicAdd(&(nodes[child_indices[child_idx]].visits), n_simulations_per_child);

    delete[] host_results;
    cudaFree(dev_positions);
    cudaFree(dev_results);
}



__global__
void simulate_position(DevicePosition *positions, int *results, int num_simulations, int NN, int seed_offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_simulations) return;

    // Initialize RNG state
    curandState state;
    curand_init(clock64() + idx + seed_offset, 0, 0, &state);

    // Copy the position to simulate
    Position current_st = positions[idx];

    int start_player = 3 - current_st.to_move;
    bool win = false;

    const int MAX_ROLLOUT_STEPS = NN;
    int steps = 0;

    // Perform simulation
    while (!current_st.is_game_over && steps < MAX_ROLLOUT_STEPS) {
        // Generate possible moves
        // For simplicity, assume a fixed maximum number of moves
        int moves[NN + 1];
        int num_moves = 0;
        for(int fc = 0; fc < NN; fc++) {
            if (is_legal_move(current_st, fc, current_st.to_move)) {
                moves[num_moves++] = fc;
            }
        }
        if(current_st.empty_spaces.size() < (size_t)(NN / 2)) {
            moves[num_moves++] = NN; // 'pass' move
        }

        if (num_moves == 0) break; // No moves available

        // Select a random move
        int move_idx = curand(state) % num_moves;
        int selected_move = moves[move_idx];

        // Apply the move
        current_st = play_move(current_st, selected_move);
        steps++;
    }

    double sc = final_score(current_st);
    if (start_player == BLACK && sc > 0.0) {
        win = true;
    } else if (start_player == WHITE && sc < 0.0) {
        win = true;
    }

    results[idx] = win ? 1 : 0;
}


void backprop(Node *node, int result, int sum_n_simulations) {
    while(node) {
        node->visits += sum_n_simulations;
        node->wins   += result;
        node = node->parent;
        result = sum_n_simulations - result;
    }
}

void mcts_iteration(Node *root, int n_simulations) {
    // 1) Selection
    Node *node = root;
    while(!node->children.empty()) {
        node = select_child(node);
    }
    expand(node);
    
    sn result = simulate_node(node, n_simulations);
    backprop(node, sim_result.wins, sim_result.sum_n_simulations);
}

int best_move(Node *root) {
    int best_fc = -1;
    double best_ratio = -1e9;
    for(auto &cptr : root->children) {
        if(cptr->visits <= 0) continue;
        double ratio = cptr->wins / (double)cptr->visits;
        if(ratio > best_ratio) {
            best_ratio = ratio;
            best_fc = cptr->move_fc;
        }
    }
    return best_fc;
}
