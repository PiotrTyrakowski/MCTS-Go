#include "constants.hpp"
#include <array>
#include <utility>


// Flatten 2D coordinate (row, col) into 1D index
constexpr int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten 1D index -> (row, col)
constexpr std::pair<int, int> unflatten(int idx) {
    return { idx / N, idx % N };
}

// Check if a (row, col) is on board
constexpr bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// Structure to hold neighbors for each cell
struct NeighborList {
    std::array<int, 4> neighbors{};  // Maximum 4 neighbors
    int count = 0;                    // Actual number of neighbors

    // constexpr constructor to allow compile-time initialization
    constexpr NeighborList() = default;

    constexpr void add_neighbor(int idx) {
        neighbors[count++] = idx;
    }
};

// constexpr function to create NeighborList for a given cell
constexpr NeighborList make_neighbor(int row, int col) {
    NeighborList list;
    if (is_on_board(row - 1, col)) {            // Up
        list.add_neighbor(flatten(row - 1, col));
    }
    if (is_on_board(row + 1, col)) {            // Down
        list.add_neighbor(flatten(row + 1, col));
    }
    if (is_on_board(row, col - 1)) {            // Left
        list.add_neighbor(flatten(row, col - 1));
    }
    if (is_on_board(row, col + 1)) {            // Right
        list.add_neighbor(flatten(row, col + 1));
    }
    return list;
}

// constexpr function to build the neighbors array
constexpr std::array<NeighborList, NN> build_neighbors_array() {
    std::array<NeighborList, NN> neighbors = {};

    // Iterate over each cell to populate its neighbors
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            neighbors[flatten(row, col)] = make_neighbor(row, col);
        }
    }

    return neighbors;
}

// Global compile-time neighbors array
constexpr auto NEIGHBORS = build_neighbors_array();

