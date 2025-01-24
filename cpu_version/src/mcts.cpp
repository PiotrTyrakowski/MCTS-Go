

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

#include "position.hpp"
// #include "neighbors.hpp"
#include "mcts.hpp"



double ucb_for_child(const Node &child, int total_visits) {
    if(child.visits == 0) {
        // Infinity in practice
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
        double u = ucb_for_child(*cptr, node->visits);
        if(u > best_value) {
            best_value = u;
            best_child = cptr.get();
        }
    }
    return best_child;
}




// typedef struct sn {
//     int wins;
//     int sum_n_simulations;
// } sn;

sn simulate_node(Node *node, int n_simulations) {
    sn result;
    result.wins = 0;
    result.sum_n_simulations = 0;

    // Example: For each child, run 5 playouts, accumulate total
    // If you'd rather simulate from the node itself, adjust accordingly.

    // std::cout << "legalmoves" << node->children.size() << '\n';

    if(node->children.size() == 0)
    {
        int w = simulate_position(node->state, NN*n_simulations);
        result.wins += w;
        result.sum_n_simulations += NN*n_simulations;
        return result;
    }

    for (auto &child : node->children) {
        int w = simulate_position(child->state, n_simulations);

        child->visits += n_simulations;
        child->wins += w;
        result.wins += n_simulations - w;
        result.sum_n_simulations += n_simulations;
    }

    return result;
}


int simulate_position(Position st, int n_simulations) {
    // Which color began this simulation?
    int start_player = 3 - st.to_move;
    
    // Count how many times "start_player" ended up winning
    int wins_for_starter = 0;

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int sim = 0; sim < n_simulations; sim++) {
        // We'll do a fresh copy for each playout
        Position current_st = st;
        const int MAX_ROLLOUT_STEPS = NN;
        // const int MAX_ROLLOUT_STEPS = 5;

        int idx = static_cast<int>(dist(rng) * PRIME);


        int steps = 0;
        while (!current_st.is_game_over && steps < MAX_ROLLOUT_STEPS) {
            // Build a naive list of candidate moves
            std::vector<int> moves(current_st.empty_spaces.begin(), current_st.empty_spaces.end());

            // if(current_st.pass_happened) {
            //     std::cout <<"something" << '\n';
            //     current_st.print();
            //     std::cout <<"moves" << moves.size() << '\n';
            // }
            
            // Optionally allow a pass if the board is relatively full
            if (current_st.empty_spaces.size() < NN / 2) {
                moves.push_back(NN);  // pass
            }


            // std::cout << "legalmoves" << moves.size() << '\n';


            // if(current_st.pass_happened){
            //     // 50% chance to pass
            //     if(dist(rng) < 0.1){
            //         current_st = play_move(current_st, NN);
            //         break;
            //     }
            // }

            // Try up to 10 random picks from 'moves' to find a legal one
            bool move_found = false;
            for (int tries = 0; tries < 10; tries++) {
                // int idx = static_cast<int>(dist(rng) * moves.size());
                idx = (idx + PRIME) % moves.size();
                int move_fc = moves[idx];

                // Check legality or pass
                if (move_fc == NN || is_legal_move(current_st, move_fc, current_st.to_move)) {
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

    }
    return wins_for_starter;
}

void backprop(Node *node, int result, int sum_n_simulations) {
    while(node) {
        node->visits += sum_n_simulations;
        node->wins   += result;
        node = node->parent;
        // Flip result so that each parent sees from their perspective
        result = sum_n_simulations - result; 
    }
}

void expand(Node *node) {
    // If no unexpanded moves, just return node
    if(node->state.is_game_over == true) {
        return;
    }

    int next_player = node->state.to_move;
    // Pick an unexpanded move
    for(auto move_fc: node->legal_moves)
    {
        Position new_state = play_move(node->state, move_fc);
        node->children.push_back(std::make_unique<Node>(new_state, node, move_fc, node->move_number + 1, next_player));
    }

    
    // std::cout << "dup2a" << node->children.size() << "\n"; // Debug output

    
}


void mcts_iteration(Node *root, int n_simulations) {
    // 1. Selection: descend until we reach a node that is not fully expanded
    Node *node = root;
    // If #children == #legal_moves, that node is fully expanded
    while(!node->children.empty() ) {
        node = select_child(node);
        // node could become a deeper child
    }

    // 2. Expansion: expand the chosen node if possible
    expand(node);

  

    // 3. Simulation: run some number of random playouts from this node
    sn sim_result = simulate_node(node, n_simulations);

    // if(sim_result.sum_n_simulations > 0)
    //     std::cout << "du3pa" << sim_result.sum_n_simulations << "\n"; // Debug output

    // 4. Backprop: update stats up the tree
    backprop(node, sim_result.wins, sim_result.sum_n_simulations);
}


int best_move(Node *root) {
    // We pick the move with the highest visitation count
    int best_fc = -1;
    int best_visits = -1;
    int best_wins = -1;
    double best_ratio = -1.0;
    for(auto &cptr : root->children) {

     

        double ratio = double(cptr->wins) / double(cptr->visits);
        // choose the best winratio instead
        if (ratio > best_ratio)
        {
            best_ratio = ratio;
            best_fc = cptr->move_fc;
        }


    }
    return best_fc;
}


