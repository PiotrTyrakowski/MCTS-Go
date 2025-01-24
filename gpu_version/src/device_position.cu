#include "device_position.cuh"
#include <stdio.h>

__constant__ NeighborList d_NEIGHBORS[NN];

void initialize_device_neighbors(const NeighborList h_neighbors[NN]) {
    cudaMemcpyToSymbol(d_NEIGHBORS, h_neighbors, sizeof(NeighborList) * NN);
}

__device__ DevicePosition::DevicePosition()
    : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) 
{
    for(int i = 0; i < NN; i++) {
        board[i] = EMPTY;
    }
}

__device__ int DevicePosition::swap_color(int color) const {
    return (color == BLACK) ? WHITE : BLACK;
}

__device__ void DevicePosition::bulk_remove_stones(const int* stones, int num_stones) {
    for(int i = 0; i < num_stones; i++) {
        int fc = stones[i];
        board[fc] = EMPTY;
    }
}

__device__ void DevicePosition::find_reached(int start, int* chain, int* reached, int& chain_size, int& reached_size) const {
    int color = board[start];
    bool visited[NN] = {false};
    int front = 0, back = 0;

    chain[back++] = start;
    visited[start] = true;
    chain_size = 1;
    reached_size = 0;

    while(front < back){
        int current = chain[front++];
        NeighborList nbrList = d_NEIGHBORS[current];
        for(int i = 0; i < nbrList.count; i++) {
            int nb = nbrList.neighbors[i];
            if(board[nb] == color && !visited[nb]) {
                visited[nb] = true;
                chain[back++] = nb;
                chain_size++;
            }
            else if(board[nb] != color && board[nb] != EMPTY) {
                reached[reached_size++] = nb;
            }
        }
    }
}

__device__ bool DevicePosition::is_legal_move(int fc, int color) {
    if(fc < 0 || fc >= NN) return false;
    if(board[fc] != EMPTY) return false;
    if(fc == ko) return false;

    DevicePosition temp = *this;
    temp.board[fc] = color;

    int opp = swap_color(color);
    int chain[NN];
    int reached[NN];
    int chain_size, reached_size;

    NeighborList nbrList = d_NEIGHBORS[fc];
    bool capture = false;
    for(int i = 0; i < nbrList.count; i++) {
        int nb = nbrList.neighbors[i];
        if(temp.board[nb] == opp) {
            temp.find_reached(nb, chain, reached, chain_size, reached_size);
            bool has_liberty = false;
            for(int j = 0; j < reached_size; j++) {
                if(temp.board[reached[j]] == EMPTY) {
                    has_liberty = true;
                    break;
                }
            }
            if(!has_liberty) {
                capture = true;
            }
        }
    }

    temp.find_reached(fc, chain, reached, chain_size, reached_size);
    bool has_liberty = false;
    for(int j = 0; j < reached_size; j++) {
        if(temp.board[reached[j]] == EMPTY) {
            has_liberty = true;
            break;
        }
    }

    if(!has_liberty && !capture) {
        return false; 
    }


    return true;
}

__device__ void DevicePosition::play_move(int fc) {
    int color = to_move;

    if(fc == NN) { 
        to_move = swap_color(color);
        if(pass_happened) {
            is_game_over = true;
            return;
        }
        pass_happened = true;
        return;
    }

    pass_happened = false;
    ko = -1;
    board[fc] = color;

    int opp_color = swap_color(color);
    int maybe_ko_checker = -1;
    int total_opp_captured = 0;

    int chain[NN];
    int reached[NN];
    int chain_size, reached_size;

    NeighborList nbrList = d_NEIGHBORS[fc];
    for(int i = 0; i < nbrList.count; i++) {
        int nb = nbrList.neighbors[i];
        if(board[nb] == opp_color) {
            find_reached(nb, chain, reached, chain_size, reached_size);
            bool has_liberty = false;
            for(int j = 0; j < reached_size; j++) {
                if(board[reached[j]] == EMPTY) {
                    has_liberty = true;
                    break;
                }
            }
            if(!has_liberty) {
                total_opp_captured += chain_size;
                if(chain_size == 1) {
                    maybe_ko_checker = nb;
                }
                bulk_remove_stones(chain, chain_size);
            }
        }
    }

    if(total_opp_captured == 1 && maybe_ko_checker != -1) {
        ko = maybe_ko_checker;
    }

    to_move = opp_color;
}

__device__ double DevicePosition::final_score() const {
    int black_count = 0, white_count = 0;
    for(int i = 0; i < NN; i++) {
        if(board[i] == BLACK) black_count++;
        if(board[i] == WHITE) white_count++;
    }
    double score = (double)black_count - (double)white_count;
    score -= KOMI;
    return score;
}

__device__ void DevicePosition::print() const {
    for(int r = 0; r < N; r++) {
        for(int c = 0; c < N; c++) {
            int fc = r * N + c;
            if(board[fc] == EMPTY)      printf(".");
            else if(board[fc] == BLACK) printf("X");
            else if(board[fc] == WHITE) printf("O");
        }
        printf("\n");
    }
    printf("Ko: %d, to_move: %s, pass_happened: %d, is_game_over: %d\n", 
           ko, (to_move == BLACK ? "BLACK" : "WHITE"), 
           pass_happened, is_game_over);
}

__global__ void initialize_positions(DevicePosition* positions, int num_positions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num_positions) {
        positions[idx] = DevicePosition();
    }
}
