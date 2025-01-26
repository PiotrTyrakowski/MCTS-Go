#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "cuda_defs.hpp"

// Board dimension constants
constexpr int N   = 5;            // Board dimension
constexpr int NN  = N * N;        // Number of intersections
constexpr int EMPTY = 0;
constexpr int BLACK = 1;
constexpr int WHITE = 2;

constexpr int PRIME = 677;

// Komi
constexpr double KOMI = 5.5;

// For MCTS
constexpr int MCTS_SIMULATIONS = 1000;
constexpr double UCB_C = 1.41421;

#endif // CONSTANTS_H
