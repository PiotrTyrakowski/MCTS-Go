

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



static std::mt19937 rng((unsigned)std::time(nullptr));


// Simple Node for MCTS
struct Node {
    // Parent
    Node *parent;


    Position state;
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
    


   
    Node(const Position &st, Node *p, int m, int move_number, int color_of_move)
        : state(st), move_fc(m), move_number(move_number), color_of_move(color_of_move), wins(0), visits(0), ucb_value(0), parent(p)
    {



        // Generate all possible moves
        for(int fc=0; fc<NN; fc++){
            if(is_legal_move(state, fc, state.to_move)) {
                legal_moves.push_back(fc);
                
            }
        }

        legal_moves.push_back(NN);

        
        // std::cout << "aaaa" << legal_moves.size() << '\n';
    }
};


double ucb_for_child(const Node &child, int total_visits);


Node* select_child(Node *node);

void expand(Node *node);

typedef struct sn {
    int wins;
    int sum_n_simulations;
} sn;

sn simulate_node(Node *node, int n_simulations);


int simulate_position(Position st, int n_simulations);

void backprop(Node *node, int result, int sum_n_simulations);




void mcts_iteration(Node *root, int n_interations);


int best_move(Node *root) ;

