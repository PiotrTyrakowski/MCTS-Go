#ifndef NEIGHBORS_H
#define NEIGHBORS_H

#include "types.cuh"
#include "cuda_defs.hpp"

// Flatten 2D coordinate (row, col) into 1D index
HOSTDEV inline constexpr int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten 1D index -> (row, col)
HOSTDEV inline constexpr IntPair unflatten(int idx) {
    return IntPair(idx / N, idx % N);
}

// Check if a (row, col) is on board
HOSTDEV inline constexpr bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// Create neighbor list for a single cell
HOSTDEV inline constexpr Array4Neighbors make_neighbor(int row, int col) {
    Array4Neighbors list;
    if (is_on_board(row - 1, col)) {            // Up
        list.push_back(flatten(row - 1, col));
    }
    if (is_on_board(row + 1, col)) {            // Down
        list.push_back(flatten(row + 1, col));
    }
    if (is_on_board(row, col - 1)) {            // Left
        list.push_back(flatten(row, col - 1));
    }
    if (is_on_board(row, col + 1)) {            // Right
        list.push_back(flatten(row, col + 1));
    }
    return list;
}

// Build the entire neighbors array
HOSTDEV inline constexpr void build_neighbors_array(Array4Neighbors* neighbors) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            neighbors[flatten(row, col)] = make_neighbor(row, col);
        }
    }
}

// Declare constant memory array
__managed__ Array4Neighbors NEIGHBORS[NN];

void initialize_neighbors_constant();

#endif // NEIGHBORS_H
