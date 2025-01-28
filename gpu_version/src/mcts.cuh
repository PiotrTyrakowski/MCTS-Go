

#include "position.cuh"





// Simple Node for MCTS
struct Node {
    // Parent
    Node *parent;


    Position state;
    // Move that led to this node (fc). For root node, move_fc = -1
    int move_fc;
    // id of the node from parent perspective
    int id;
    int move_number;
    int color_of_move;
    // Statistics

    int wins;
    int visits;
    double ucb_value;

    double best_child_ucb_value;
    int best_child_id;



    // Children
    bool expanded;
    Node** children;
    ArrayInt legal_moves;
    Array4Neighbors* neighbors_array;


   
    HOSTDEV Node(const Position &st, Node *p, int move_fc, int move_number, int color_of_move, int id, Array4Neighbors* neighbors_array)
        : parent(p), state(st), move_fc(move_fc), id(id), move_number(move_number),
          color_of_move(color_of_move), wins(0.0), visits(0), ucb_value(0.0),
          best_child_ucb_value(-1e9), best_child_id(-1),
          expanded(false), children(nullptr), neighbors_array(neighbors_array)
    {

        // Generate all possible moves
        for(int fc=0; fc<=NN; fc++){
            if(is_legal_move(state, fc, state.to_move, neighbors_array)) {
                legal_moves.push_back(fc);

                
            }
        }

        // alocate chldren to be size of legal_moves.size() (i want it  to be array)
        
    }

    HOSTDEV Node() {}


    HOSTDEV ~Node() {
        if (children != nullptr) {
            for (int i = 0; i < legal_moves.size(); ++i) {
                if (children[i] != nullptr) {
                    delete children[i]; // Recursively delete children
                }
            }
            delete[] children; // Delete the array of child pointers
        }
    }
};


HOSTDEV double ucb_for_child(const Node &child, int total_visits);


HOSTDEV Node* select_child(Node *node);

void expand(Node *node);



__global__ void simulate_node(Position* children_positions, int n_children, int* wins, int* simulations, Array4Neighbors* neighbors);


HOSTDEV int calculate_win_points(int start_player, double sc);

__device__ void simulate_position(Position st, int* wins, int tid,  Array4Neighbors* neighbors_array);


HOSTDEV void backprop(Node *node, int result, int sum_n_simulations);




__host__ void mcts_iteration(Node *root, int n_interations);


// first is best move second is child id
HOSTDEV int find_best_child(Node *root) ;

