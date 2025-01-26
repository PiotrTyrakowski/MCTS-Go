



#include "position.cuh"
#include "mcts.cuh"
#include "cmath"



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




HOSTDEV void simulate_node(Node *node, int n_simulations, int* wins, int* sum_n_simulations) {
    *wins = 0;
    *sum_n_simulations = 0;

    // here we will make cuda loading
    Array4Neighbors* neighbors_array = node->neighbors_array;

    if(node->legal_moves.size() == 0)
    {
        int w = 0;
        simulate_position(node->state, NN*n_simulations, &w, neighbors_array);
        *wins += w;
        *sum_n_simulations += NN*n_simulations;
        return;
    }

    double best_child_ucb_value = -1e9;
    int best_child_id = -1;
    int n_legal_moves = node->legal_moves.size();
    *sum_n_simulations = n_legal_moves * n_simulations;

    for (int i = 0; i < n_legal_moves; i++) {
        // here i want to mopdify child (so maybe reference idk?)
        Node* child = node->children[i];
        int w = 0;
        simulate_position(child->state, n_simulations, &w, neighbors_array);

        child->visits += n_simulations;
        child->wins += w;
        *wins += n_simulations - w;

        double child_ucb_value = ucb_for_child(*child, *sum_n_simulations);

        // std::cout << child_ucb_value << "fdsfsd\n";
        

        if(child_ucb_value > best_child_ucb_value)
        {
            best_child_id = i;
            best_child_ucb_value = child_ucb_value;
        }
        
    }

    node->best_child_id = best_child_id;
    node->best_child_ucb_value = best_child_ucb_value;

    return;
}


HOSTDEV void simulate_position(Position st, int n_simulations, int* wins, Array4Neighbors* neighbors_array) {
    // Which color began this simulation?
    int start_player = swap_color(st.to_move);
    
    // Count how many times "start_player" ended up winning
    int wins_for_starter = 0;

    int idx = 5 + n_simulations;

    // here will get some random move

    for (int sim = 0; sim < n_simulations; sim++) {
        // We'll do a fresh copy for each playout
        Position current_st = st;
        const int MAX_ROLLOUT_STEPS = NN;
        
        // TODO: change this 
        // int idx = 5; // i will change this leter to be random with some funciton don't worry about it.



        int steps = 0;
        while (!current_st.is_game_over && steps < MAX_ROLLOUT_STEPS) {
            // Build a naive list of candidate moves
            ArrayInt moves = current_st.empty_spaces.set; 

            // Optionally allow a pass if the board is relatively full
            if (current_st.empty_spaces.size() < NN / 2) {
                moves.push_back(NN);  // pass
            }

           
            bool move_found = false;
            for (int tries = 0; tries < 10; tries++) {
                idx = (idx + PRIME) % moves.size();
                int move_fc = moves[idx];

                // Check legality or pass
                if (move_fc == NN || is_legal_move(current_st, move_fc, current_st.to_move, neighbors_array)) {
                    current_st = play_move(current_st, move_fc, neighbors_array);
                    move_found = true;
                    break;
                }
            }

            // If we still couldn't find anything, just pass:
            if(!move_found) {
                current_st = play_move(current_st, NN, neighbors_array);
            }

            steps++;
        }

        // Game might have ended or we hit the rollout limit
        double sc = final_score(current_st, neighbors_array); // Black - White - Komi
        // If sc > 0 => Black is ahead; if sc < 0 => White is ahead
        bool black_is_winner = (sc > 0.0);
        bool white_is_winner = (sc < 0.0);

        // Convert the final board outcome to "start_player" perspective:
      
        if (start_player == BLACK && black_is_winner) {
            wins_for_starter++;
        }
        else if (start_player == WHITE && white_is_winner) {
            wins_for_starter++;
        }
        // If we want to handle draws differently, we can do so here

    }
    *wins = wins_for_starter;
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

HOSTDEV void expand(Node *node) { 
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

    node->expaned = true;
}



HOSTDEV void mcts_iteration(Node *root, int n_simulations) {
    // 1. Selection: descend until we reach a node that is not fully expanded
    Node *node = root;
    // If #children == #legal_moves, that node is fully expanded

    while(true == node->expaned) {
        node = select_child(node);
    }
    // 2. Expansion: expand the chosen node if possible
    expand(node);



    int wins = 0;
    int sum_n_simulations = 0;

    // 3. Simulation: run some number of random playouts from this node
    simulate_node(node, n_simulations, &wins, &sum_n_simulations);


    // 4. Backprop: update stats up the tree
    backprop(node, wins, sum_n_simulations);


}


HOSTDEV int best_move(Node *root) {
    // We pick the move with the highest visitation count
    int best_fc = -1;
    double best_ratio = -1.0;
    
    for(int id = 0; id <root->legal_moves.size(); id++)
    { 
        Node* cptr = root->children[id];

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


