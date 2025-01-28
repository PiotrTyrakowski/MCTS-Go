#include "types.hpp"




// Flatten 2D coordinate (row, col) into 1D index
int flatten(int row, int col);

// Unflatten 1D index -> (row, col)
IntPair unflatten(int idx);

// Check if a (row, col) is on board
bool is_on_board(int row, int col);




ArrayInt make_neighbor(int row, int col);

ArrayInt* build_neighbors_array();


