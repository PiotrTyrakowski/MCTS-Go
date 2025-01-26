#include "constants.hpp"
#include "neighbors.hpp"
#include <array>
#include <utility>




// Flatten 2D coordinate (row, col) into 1D index
int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten 1D index -> (row, col)
IntPair unflatten(int idx) {
    IntPair pair(idx / N, idx % N);
    return pair;
}

// Check if a (row, col) is on board
bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// constexpr function to create NeighborList for a given cell
ArrayInt make_neighbor(int row, int col) {
    ArrayInt list;
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

// constexpr function to build the neighbors array
ArrayInt* build_neighbors_array() {
    static ArrayInt neighbors[NN];;

    // Iterate over each cell to populate its neighbors
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            neighbors[flatten(row, col)] = make_neighbor(row, col);
        }
    }

    return neighbors;
}

