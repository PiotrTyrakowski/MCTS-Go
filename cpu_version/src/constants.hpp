#ifndef CONSTANTS_H
#define CONSTANTS_H

static const int N = 5;            // Board dimension
static const int NN = N * N;        // Number of intersections
static const int EMPTY = 0;
static const int BLACK = 1;
static const int WHITE = 2;
static const int PRIME = 677;

// Typically, standard komi in 19x19 might be 6.5-7.5
static const double KOMI = 5.5;

// For MCTS
static const int MCTS_SIMULATIONS = 1000;  // You can adjust
static const double UCB_C = 1.41421;   

#endif