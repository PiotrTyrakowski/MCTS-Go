#ifndef DEVICE_POSITION_CUH
#define DEVICE_POSITION_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>




struct DevicePosition {
    int board[NN];
    int ko;
    int to_move;
    bool pass_happened;
    bool is_game_over;

    __device__ DevicePosition();

    __device__ void print() const; 
    __device__ void bulk_remove_stones(const int* stones, int num_stones);
    __device__ void find_reached(int start, int* chain, int* reached, int& chain_size, int& reached_size) const;
    __device__ int swap_color(int color) const;
    __device__ bool is_legal_move(int fc, int color) const;
    __device__ void play_move(int fc);
    __device__ double final_score() const;
};

__global__ void initialize_positions(DevicePosition* positions, int num_positions);

#endif // DEVICE_POSITION_CUH
