#include "position.cuh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

Position::Position() 
    : ko(-1), to_move(BLACK), pass_happened(false), is_game_over(false) 
{
    board.fill(EMPTY);
    for(int i = 0; i < NN; i++) {
        empty_spaces.insert(i);
    }
}

Position::Position(const std::array<int, NN>& board_, 
                   int ko_, 
                   int to_move_, 
                   bool pass_happened_,
                   bool is_game_over_)
  : board(board_), 
    ko(ko_), 
    to_move(to_move_), 
    pass_happened(pass_happened_), 
    is_game_over(is_game_over_)
{
    empty_spaces.clear();
    for(int i = 0; i < NN; i++) {
        if(board[i] == EMPTY) {
            empty_spaces.insert(i);
        }
    }
}

void Position::print() const {
    for(int r = 0; r < N; r++){
        for(int c = 0; c < N; c++){
            int fc = r * N + c;
            if(board[fc] == EMPTY)      std::cout << ".";
            else if(board[fc] == BLACK) std::cout << "X";
            else if(board[fc] == WHITE) std::cout << "O";
        }
        std::cout << "\n";
    }
    std::cout << "Ko: " << ko 
              << ", to_move: " << (to_move == BLACK ? "BLACK" : "WHITE")
              << ", pass_happened: " << pass_happened 
              << ", is_game_over: " << is_game_over << "\n";
}


void bulk_remove_stones(Position &pos, const std::vector<int> &stones) {
    for(int fc : stones) {
        pos.board[fc] = EMPTY;
        pos.empty_spaces.insert(fc);
    }
}

std::pair<std::vector<int>, std::vector<int>> find_reached(const Position &pos, int start)
{
    int color = pos.board[start];

    std::vector<int> chain;
    chain.reserve(NN);
    std::vector<int> reached;
    reached.reserve(NN);

    std::vector<bool> visited(NN, false);
    visited[start] = true;

    std::queue<int> frontier;
    frontier.push(start);
    chain.push_back(start);

    while(!frontier.empty()){
        int current = frontier.front();
        frontier.pop();
        
        const auto &nbrList = NEIGHBORS[current];
        for(int i = 0; i < nbrList.count; i++){
            int nb = nbrList.neighbors[i];
            if(pos.board[nb] == color && !visited[nb]){
                visited[nb] = true;
                frontier.push(nb);
                chain.push_back(nb);
            }
            else if(pos.board[nb] != color) {
                reached.push_back(nb);
            }
        }
    }
    return {chain, reached};
}

std::vector<int> maybe_capture_stones(Position &pos, int fc) {
    auto [chain, reached] = find_reached(pos, fc);

    bool has_liberty = false;
    for(int r : reached) {
        if(pos.board[r] == EMPTY) {
            has_liberty = true;
            break;
        }
    }
    if(!has_liberty) {
        bulk_remove_stones(pos, chain);
        return chain;
    }
    return {};
}

bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone) {
    if(pos.board[maybe_ko_checker] != EMPTY) return false;
    if(pos.board[played_stone] == EMPTY) return false;

    int played_stone_color = pos.board[played_stone];
    
    const auto &nbrList = NEIGHBORS[played_stone];
    for(int i = 0; i < nbrList.count; i++) {
        int nb = nbrList.neighbors[i];
        if(pos.board[nb] == played_stone_color) {
            return false;
        }
        if(pos.board[nb] == EMPTY && nb != maybe_ko_checker) {
            return false;
        }
    }
    return true;
}

// Check if move at fc is legal
bool is_legal_move(const Position &pos, int fc, int color) {
    if(fc < 0 || fc > NN) return false; 
    if(fc < NN && pos.board[fc] != EMPTY) return false; 
    if(fc == pos.ko) return false;   
    if(fc == NN) return true;         

    // Make copy
    Position temp = pos;
    temp.board[fc] = color;

    int opp = swap_color(color);
    const auto &nbrList = NEIGHBORS[fc];
    for(int i = 0; i < nbrList.count; i++){
        int nb = nbrList.neighbors[i];
        if(temp.board[nb] == opp) {
            maybe_capture_stones(temp, nb);
        }
    }


    auto captured = maybe_capture_stones(temp, fc);
    if(!captured.empty()) {
     
        return false;
    }
    return true;
}

// Play a move (assumes move is legal)
Position play_move(const Position &oldPos, int fc) {
    Position pos = oldPos;
    int color = pos.to_move;


    if(fc == NN) {
        pos.to_move = swap_color(color);
        if(pos.pass_happened) {
            pos.is_game_over = true;
            return pos;
        }
        pos.pass_happened = true;
        return pos;
    }


    pos.pass_happened = false;
    pos.ko = -1;
    pos.board[fc] = color;
    pos.empty_spaces.erase(fc);

    int opp_color = swap_color(color);

    int maybe_ko_checker = -1;
    int total_opp_captured = 0;

    const auto &nbrList = NEIGHBORS[fc];
    for(int i = 0; i < nbrList.count; i++){
        int nb = nbrList.neighbors[i];
        if(pos.board[nb] == opp_color) {
            auto captured = maybe_capture_stones(pos, nb);
            total_opp_captured += (int)captured.size();
            if(captured.size() == 1) {
                maybe_ko_checker = nb;
            }
        }
    }


    if(total_opp_captured == 1 && 
       is_koish_for_next_player(pos, maybe_ko_checker, fc)) 
    {
        pos.ko = maybe_ko_checker;
    }

    pos.to_move = opp_color;
    return pos;
}

double final_score(const Position &pos) {

    Position temp = pos;

    for(int i = 0; i < NN; i++){
        if(temp.board[i] == EMPTY) {
            auto [chain, reached] = find_reached(temp, i);
            int candidate = -1;
            bool mixed = false;
            for(int r: reached) {
                if(temp.board[r] == BLACK || temp.board[r] == WHITE) {
                    if(candidate < 0) candidate = temp.board[r];
                    else if(temp.board[r] != candidate) {
                        mixed = true;
                        break;
                    }
                }
            }
            if(!mixed && candidate > 0) {

                for(int fc : chain) {
                    temp.board[fc] = candidate;
                }
            } else {
      
                for(int fc : chain) {
                    temp.board[fc] = -1; 
                }
            }
        }
    }

    int black_count = 0, white_count = 0;
    for(int i = 0; i < NN; i++){
        if(temp.board[i] == BLACK) black_count++;
        if(temp.board[i] == WHITE) white_count++;
    }
    double score = (double)black_count - (double)white_count;
    score -= KOMI;
    return score;
}
