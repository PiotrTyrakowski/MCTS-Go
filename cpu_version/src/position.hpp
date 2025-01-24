#ifndef POSITION_HPP
#define POSITION_HPP

#include "neighbors.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>

// ... existing code for swap_color inline function ...

////////////////////////////////////////////////////
// Data structure representing the game state
// //////////////////////////////////////////////////

inline int swap_color(int color);
   



struct Position {
    // Board array, 1D, each entry ∈ {EMPTY, BLACK, WH`ITE}
    std::array<int, NN> board;
    // Ko point (if any); -1 means no Ko
    int ko;
    // Next player to move
    int to_move;
    
    bool pass_happened;

    bool is_game_over;

    std::unordered_set<int> empty_spaces;

    Position() : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) {
        board.fill(EMPTY);
        for(int i = 0; i < NN; i++) {
            empty_spaces.insert(i);
        }
    }

    Position(const std::array<int, NN>& board_, int ko_, int to_move_, 
             bool pass_happened_, bool is_game_over_)
        : board(board_), ko(ko_), to_move(to_move_), 
          pass_happened(pass_happened_), is_game_over(is_game_over_) 
    {
        // Initialize empty_spaces based on the provided board
        empty_spaces.clear();
        for(int i = 0; i < NN; i++) {
            if(board[i] == EMPTY) {
                empty_spaces.insert(i);
            }
        }
    }




    // Print board to stdout (for debugging)
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


};

// Function declarations
void bulk_remove_stones(Position &pos, const std::vector<int> &stones);

std::pair<std::vector<int>, std::vector<int>> find_reached(
    const Position &pos, int start);

std::vector<int> maybe_capture_stones(Position &pos, int fc);

bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone);

bool is_legal_move(const Position &pos, int fc, int color);

Position play_move(const Position &oldPos, int fc);

double final_score(const Position &pos);

#endif // POSITION_HPP