#ifndef NEIGHBORS_CUH
#define NEIGHBORS_CUH

#include "constants.cuh"
#include <array>
#include <utility>


HD constexpr int flatten(int row, int col) {
    return row * N + col;
}

HD constexpr std::pair<int, int> unflatten(int idx) {
    return { idx / N, idx % N };
}


HD constexpr bool is_on_board(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

struct NeighborList {
    std::array<int, 4> neighbors{};  
    int count = 0;                   

    constexpr NeighborList() = default;

    HD constexpr void add_neighbor(int idx) {
        neighbors[count++] = idx;
    }
};

constexpr NeighborList make_neighbor(int row, int col) {
    NeighborList list;
    if (is_on_board(row - 1, col)) {
        list.add_neighbor(flatten(row - 1, col)); // Up
    }
    if (is_on_board(row + 1, col)) {
        list.add_neighbor(flatten(row + 1, col)); // Down
    }
    if (is_on_board(row, col - 1)) {
        list.add_neighbor(flatten(row, col - 1)); // Left
    }
    if (is_on_board(row, col + 1)) {
        list.add_neighbor(flatten(row, col + 1)); // Right
    }
    return list;
}

constexpr std::array<NeighborList, NN> build_neighbors_array() {
    std::array<NeighborList, NN> neighbors = {};

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            neighbors[flatten(row, col)] = make_neighbor(row, col);
        }
    }
    return neighbors;
}

constexpr auto NEIGHBORS = build_neighbors_array();

#endif
