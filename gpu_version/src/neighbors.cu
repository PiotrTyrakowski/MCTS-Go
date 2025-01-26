#include "neighbors.cuh"
#include "constants.cuh"

void initialize_neighbors_constant() {
    Array4Neighbors host_NEIGHBORS[NN];
    build_neighbors_array(host_NEIGHBORS);
    cudaMemcpyToSymbol(NEIGHBORS, host_NEIGHBORS, sizeof(host_NEIGHBORS));
}