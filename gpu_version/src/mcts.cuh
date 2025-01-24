// mcts_simulation.cuh
#ifndef MCTS_SIMULATION_CUH
#define MCTS_SIMULATION_CUH

#include "neighbors.cuh"

Node::Node(const Position &st, Node *p, int m, int move_num, int color_of_move);

double ucb_for_child(const Node &child, int total_visits);

Node* select_child(Node *node);

void expand(Node *node);



void simulate_node(Node *nodes, int n_simulations_per_child) ;



__global__
void simulate_position(Position *positions, int *results, int num_simulations, int NN, int seed_offset);


void backprop(Node *node, int result, int sum_n_simulations);

void mcts_iteration(Node *root, int n_simulations);

int best_move(Node *root);