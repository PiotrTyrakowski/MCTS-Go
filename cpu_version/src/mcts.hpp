
#include <random>
#include <ctime>

#include "position.hpp"



static std::mt19937 rng((unsigned)std::time(nullptr));


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
    


   
    Node(const Position &st, Node *p, int move_fc, int move_number, int color_of_move, int id)
        : parent(p), state(st), move_fc(move_fc), id(id), move_number(move_number),
          color_of_move(color_of_move), wins(0.0), visits(0), ucb_value(0.0),
          best_child_ucb_value(-1e9), best_child_id(-1),
          expaned(false), children(nullptr)
    {

        // Generate all possible moves
        for(int fc=0; fc<=NN; fc++){
            if(is_legal_move(state, fc, state.to_move)) {
                legal_moves.push_back(fc);

                
            }
        }

        // alocate chldren to be size of legal_moves.size() (i want it  to be array)
        
    }
};


double ucb_for_child(const Node &child, int total_visits);


Node* select_child(Node *node);

void expand(Node *node);

// typedef struct sn {
//     int wins;
//     int sum_n_simulations;
// } sn;

// sn 

void simulate_node(Node *node, int n_simulations, int* wins, int* sum_n_simulations);


// int
void simulate_position(Position st, int n_simulations, int* wins);

void backprop(Node *node, int result, int sum_n_simulations);




void mcts_iteration(Node *root, int n_interations);


int best_move(Node *root) ;

