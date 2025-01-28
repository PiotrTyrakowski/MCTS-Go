#include "position.cuh"
#include "mcts.cuh"
#include "utils.cuh"
#include "cmath"

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

HOSTDEV double ucb_for_child(const Node &child, int total_visits) {
    if(child.visits == 0) {
        // Infinity in practice
        return 1e9;
    }
    double exploitation = child.wins / (double)child.visits;
    // replace this with some cuda frineds sqrt algorithm and log algorith. i donn't want to use std at all.
    double exploration  = UCB_C * sqrt(log((double)total_visits) / (double)child.visits);
    return exploitation + exploration;
}

HOSTDEV Node* select_child(Node *node) {
    Node* best_child = nullptr;
    double best_value = -1e9;
    for(int i = 0; i < node->legal_moves.size(); i++) {
        Node* cptr = node->children[i];
        double u = ucb_for_child(*cptr, node->visits);
        if(u > best_value) {
            best_value = u;
            best_child = cptr;
        }
    }
    return best_child;
}

__global__ void simulate_node(Position* children_positions, int n_children, int* wins, int* simulations, Array4Neighbors* neighbors) {
    __shared__ Array4Neighbors* shared_neighbors; 
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        shared_neighbors = neighbors;
    }
    
    __syncthreads();  // Add synchronization here


    int kid_id = tid % n_children;

    
    int w = 0;
    simulate_position(children_positions[kid_id], &w, tid, shared_neighbors);

    atomicAdd(&wins[kid_id], w);
    atomicAdd(&simulations[kid_id], 1);

    __syncthreads();

    return;
}



HOSTDEV int calculate_win_points(int start_player, double sc)
{   
    bool black_is_winner = (sc > 0.0);
    bool white_is_winner = (sc < 0.0);

    if (start_player == BLACK && black_is_winner) {
        return 1;
    }
    else if (start_player == WHITE && white_is_winner) {
        return 1;
    }
    return 0;
}

__device__ void simulate_position(Position st, int* wins, int tid,  Array4Neighbors* neighbors_array) {
    // Which color began this simulation?
    int start_player = swap_color(st.to_move);
    
    int idx = tid;


    int rand_number = tid;

    Position current_st = st;
    const int MAX_ROLLOUT_STEPS = NN / 4 + current_st.empty_spaces.size();
        
    int steps = 0;
    while (!current_st.is_game_over && steps < MAX_ROLLOUT_STEPS) {
        // Build a naive list of candidate moves
        ArrayInt moves = current_st.empty_spaces.set; 

        // Optionally allow a pass if the board is relatively full
        if (current_st.empty_spaces.size() < NN / 2) {
            moves.push_back(NN);  // pass
        }
        
        rand_number = hash(rand_number + SEED);

        idx = (idx + rand_number) % moves.size();

        
        bool move_found = false;
        for (int tries = 0; tries < 10; tries++) {
            int move_fc = moves[idx];

            // Check legality or pass
            if (move_fc == NN || is_legal_move(current_st, move_fc, current_st.to_move, neighbors_array)) {
                current_st = play_move(current_st, move_fc, neighbors_array);
                move_found = true;
                break;
            }

            idx = (idx + PRIME) % moves.size();
        }

        // If we still couldn't find anything, just pass:
        if(!move_found) {
            current_st = play_move(current_st, NN, neighbors_array);
        }

        steps++;
    }

    double sc = final_score(current_st, neighbors_array); // Black - White - Komi
        
    *wins = calculate_win_points(start_player, sc);
}

HOSTDEV void backprop(Node *node, int result, int sum_n_simulations) {
    while(nullptr != node) {
        // std::cout << "hehe " << '\n';

        node->visits += sum_n_simulations;
        node->wins   += result;

        Node* parent = node->parent;
        if(nullptr == parent) {
            break; // We reached root
        }
        int parent_visits = parent->visits + sum_n_simulations;

 
        double node_ucb_value = ucb_for_child(*node, parent_visits);





        // ucb_for_child
        // Flip result so that each parent sees from their perspective
        node = parent;
        result = sum_n_simulations - result; 
    }
}

void expand(Node *node) { 
    if (node->state.is_game_over) {
        return; // no children to expand if game is over
    }
    int n_childs = node->legal_moves.size();
    // Allocate the array of child pointers only once
    node->children = new Node*[n_childs];

    // For each legal move, create a child node
    int next_player = node->state.to_move;
    for(int id = 0; id < n_childs; id++) {
        int move_fc = node->legal_moves[id];
        Position new_state = play_move(node->state, move_fc, node->neighbors_array);
        node->children[id] = new Node(new_state, node, move_fc,
                                      node->move_number + 1,
                                      next_player, 
                                      id,
                                      node->neighbors_array);
    }

    node->expanded = true;
}



__host__ void mcts_iteration(Node *root, int n_simulations) {
    // 1. Selection: descend until we reach a node that is not fully expanded
    Node *node = root;
    // If #children == #legal_moves, that node is fully expanded

    while(true == node->expanded) {
        node = select_child(node);
    }
    // 2. Expansion: expand the chosen node if possible
    expand(node);

    int n_children = node->legal_moves.size();
    int wins = 0;
    int sum_n_simulations = 0;

    if(n_children == 0)
    {
        int start_player = swap_color(node->state.to_move);
        double sc = final_score(node->state, node->neighbors_array);
        int small_win = calculate_win_points(start_player, sc);
        wins = small_win * n_simulations;
        sum_n_simulations = n_simulations;
    } 
    else {

        int blocksPerGrid = (n_simulations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        Position* d_children_positions;
        Position* h_children_positions = new Position[n_children];
        
        Array4Neighbors* d_neighbors;
        int* d_wins;
        int* d_simulations;

        for (int i = 0; i < n_children; i++) {
            h_children_positions[i] = node->children[i]->state;
        }


        cudaCheckError(cudaMalloc((void**)&d_children_positions, n_children * sizeof(Position)));
        cudaCheckError(cudaMemcpy(d_children_positions, h_children_positions, n_children * sizeof(Position), cudaMemcpyHostToDevice));


        cudaCheckError(cudaMalloc((void**)&d_neighbors, NN * sizeof(Array4Neighbors)));
        cudaCheckError(cudaMemcpy(d_neighbors, node->neighbors_array, NN * sizeof(Array4Neighbors), cudaMemcpyHostToDevice));


        cudaCheckError(cudaMalloc((void**)&d_wins, n_children * sizeof(int)));
        cudaCheckError(cudaMalloc((void**)&d_simulations, n_children * sizeof(int)));






        // 3. Simulation: run some number of random playouts from this node
        simulate_node<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_children_positions, n_children, d_wins, d_simulations, d_neighbors);

       

        int* h_wins = new int[n_children];
        int* h_simulations = new int[n_children];

        cudaCheckError(cudaMemcpy(h_wins, d_wins, n_children * sizeof(int), cudaMemcpyDeviceToHost));
        cudaCheckError(cudaMemcpy(h_simulations, d_simulations, n_children * sizeof(int), cudaMemcpyDeviceToHost));

        for(int i = 0; i < n_children; i++)
        {
            node->children[i]->wins = h_wins[i];
            node->children[i]->visits = h_simulations[i];
            wins += (h_simulations[i] - h_wins[i]);
            sum_n_simulations += h_simulations[i];
        }

        cudaFree(d_children_positions);
        cudaFree(d_neighbors);
        cudaFree(d_wins);
        cudaFree(d_simulations);

        delete[] h_children_positions;
        delete[] h_simulations;
        delete[] h_wins;
    

    }

    


    // 4. Backprop: update stats up the tree
    backprop(node, wins, sum_n_simulations);


}


HOSTDEV int find_best_child(Node *root) {
    double best_ratio = -1.0;
    int best_id = -1;
    
    for(int id = 0; id <root->legal_moves.size(); id++)
    { 
        Node* cptr = root->children[id];

        double ratio = double(cptr->wins) / double(cptr->visits);
        if (ratio > best_ratio)
        {
            best_ratio = ratio;
            best_id = id;
        }


    }
    return best_id;
}


