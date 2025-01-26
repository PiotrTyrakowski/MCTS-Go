#ifndef POSITION_HPP
#define POSITION_HPP


#include "neighbors.cuh"
#include <iostream>





HOSTDEV inline int swap_color(int color) {
    if(color == BLACK) return WHITE;
    if(color == WHITE) return BLACK;
    return color;
}
   

struct Position {
    // Board array, 1D, each entry ∈ {EMPTY, BLACK, WH`ITE}
    ArrayInt board;
    // Ko point (if any); -1 means no Ko
    int ko;
    // Next player to move
    int to_move;
    
    bool pass_happened;

    bool is_game_over;

    UnorderedSet empty_spaces;

    HOSTDEV Position() : board(ArrayInt(NN)), ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) {
        for(int i = 0; i < NN; i++) {
            empty_spaces.insert(i);
        }
    }

    HOSTDEV Position(const ArrayInt& board_, int ko_, int to_move_, 
             bool pass_happened_, bool is_game_over_)
        : board(board_), ko(ko_), to_move(to_move_), 
          pass_happened(pass_happened_), is_game_over(is_game_over_) 
    {
        // Initialize empty_spaces based on the provided board
        for(int i = 0; i < NN; i++) {
            if(board[i] == EMPTY) {
                empty_spaces.insert(i);
            }
        }
    }

    // Print board to stdout (for debugging)
    #ifndef __CUDA_ARCH__
        void print() const {
            for(int r=0; r<N; r++){
                for(int c=0; c<N; c++){
                    int fc = flatten(r, c);
                    if(board[fc] == EMPTY) std::cout << ".";
                    else if(board[fc] == BLACK) std::cout << "X";
                    else if(board[fc] == WHITE) std::cout << "O";
                }
                std::cout << "\n";
            }
            std::cout << "Ko: " << ko << ", to_move: " << (to_move == BLACK ? "BLACK" : "WHITE") << ", pass_happened: " << pass_happened << ", is_game_over: " << is_game_over << "\n";
        }
    #else
    // On device, just define an empty function or do nothing
        __device__ void print() const {}
    #endif

};

// Function declarations
HOSTDEV void bulk_remove_stones(Position &pos, const ArrayInt &stones);

HOSTDEV ArrayIntPair find_reached(const Position &pos, int start, Array4Neighbors* neighbors_array);

HOSTDEV ArrayInt maybe_capture_stones(Position &pos, int fc, Array4Neighbors* neighbors_array);

HOSTDEV bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone, Array4Neighbors* neighbors_array);

HOSTDEV bool is_legal_move(const Position &pos, int fc, int color, Array4Neighbors* neighbors_array);

HOSTDEV Position play_move(const Position &oldPos, int fc, Array4Neighbors* neighbors_array);


HOSTDEV double final_score(const Position &pos, Array4Neighbors* neighbors_array);

#endif // POSITION_CUH