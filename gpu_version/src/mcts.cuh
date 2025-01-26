

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

    double wins;
    int visits;
    double ucb_value;

    double best_child_ucb_value;
    int best_child_id;



    // Children
    bool expaned;
    Node** children;
    ArrayInt legal_moves;
    Array4Neighbors* neighbors_array;


   
    HOSTDEV Node(const Position &st, Node *p, int move_fc, int move_number, int color_of_move, int id, Array4Neighbors* neighbors_array)
        : parent(p), state(st), move_fc(move_fc), id(id), move_number(move_number),
          color_of_move(color_of_move), wins(0.0), visits(0), ucb_value(0.0),
          best_child_ucb_value(-1e9), best_child_id(-1),
          expaned(false), children(nullptr), neighbors_array(neighbors_array)
    {

        // Generate all possible moves
        for(int fc=0; fc<=NN; fc++){
            if(is_legal_move(state, fc, state.to_move, neighbors_array)) {
                legal_moves.push_back(fc);

                
            }
        }

        // alocate chldren to be size of legal_moves.size() (i want it  to be array)
        
    }
};


HOSTDEV double ucb_for_child(const Node &child, int total_visits);


HOSTDEV Node* select_child(Node *node);

HOSTDEV void expand(Node *node);



HOSTDEV void simulate_node(Node *node, int n_simulations, int* wins, int* sum_n_simulations);


// int
HOSTDEV void simulate_position(Position st, int n_simulations, int* wins, Array4Neighbors* neighbors_array);

HOSTDEV void backprop(Node *node, int result, int sum_n_simulations);




HOSTDEV void mcts_iteration(Node *root, int n_interations);


HOSTDEV int best_move(Node *root) ;

