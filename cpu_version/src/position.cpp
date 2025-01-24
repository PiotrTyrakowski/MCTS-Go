// #include "neighbors.hpp"
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
#include "position.hpp"


// Swap colors
inline int swap_color(int color) {
    if(color == BLACK) return WHITE;
    if(color == WHITE) return BLACK;
    return color;
}

// Helper: Bulk remove stones from board
void bulk_remove_stones(Position &pos, const std::vector<int> &stones) {
    for(int fc : stones) {
        pos.board[fc] = EMPTY;
        pos.empty_spaces.insert(fc);
    }
}

// Finds two things:
// 1. A chain: all connected stones of the same color starting from 'start'
// 2. All points adjacent to this chain (reached points)
// Returns: pair of vectors {chain, reached}
std::pair<std::vector<int>, std::vector<int>> find_reached(
    const Position &pos, int start)
{
    // Get the color of the starting point (BLACK, WHITE, or EMPTY)
    int color = pos.board[start];
    
    // Will store all connected stones of the same color
    std::vector<int> chain;
    chain.reserve(NN);  // Pre-allocate max possible size
    
    // Will store all points adjacent to the chain
    std::vector<int> reached;
    reached.reserve(NN);

    // Keep track of which points we've already processed
    std::vector<bool> visited(NN, false);
    visited[start] = true;
    
    // BFS queue for exploring connected stones
    std::queue<int> frontier;
    frontier.push(start);
    chain.push_back(start);

    // Breadth-first search through connected stones
    while(!frontier.empty()){
        int current = frontier.front();
        frontier.pop();
        
        // Check all neighboring points
        for(int i = 0; i < NEIGHBORS[current].count; i++){
            int nb = NEIGHBORS[current].neighbors[i];
            if(pos.board[nb] == color && !visited[nb]){
                // Same color and not visited: add to chain
                visited[nb] = true;
                frontier.push(nb);
                chain.push_back(nb);
            }
            else if(pos.board[nb] != color) {
                // Different color or empty: add to reached points
                reached.push_back(nb);
            }
        }
    }
    
    return {chain, reached};
}

// Attempt to capture a chain if it has no liberties
// Returns the set of captured stones (if any).
std::vector<int> maybe_capture_stones(Position &pos, int fc) {
    // Find all connected stones of the same color and their adjacent points
    auto [chain, reached] = find_reached(pos, fc);

    if (chain.size() == NN) return {};

    // Check if the chain has any liberties (empty adjacent points)
    bool has_liberty = false;
    for(int r : reached) {
        if(pos.board[r] == EMPTY) {
            has_liberty = true;
            break;
        }
    }

    // If no liberties found, capture the entire chain
    if(!has_liberty){
        bulk_remove_stones(pos, chain);
        return chain;  // Return the captured stones
    }
    return {};  // Return empty vector if no stones were captured
}

// Check if fc is "ko-ish": the move just captured exactly 1 stone
// and left a single surrounded point with no liberties.
bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone) {

    // Check if the played stone is empty
    if(pos.board[played_stone] == EMPTY) return false;

    // Check if the maybe_ko_checker is empty
    if(pos.board[maybe_ko_checker] != EMPTY) return false;


    int played_stone_color = pos.board[played_stone];

   
    for(int i = 0; i < NEIGHBORS[played_stone].count; i++) {
        int nb = NEIGHBORS[played_stone].neighbors[i];

        if(pos.board[nb] == played_stone_color) {
            return false;
        }

        if(pos.board[nb] == EMPTY && nb != maybe_ko_checker) {
            return false;
        }

    }
    return true; 
}

// Check if move at fc is legal (including ko, suicide)
bool is_legal_move(const Position &pos, int fc, int color) {
    if(fc < 0 || fc > NN) return false;
    if(pos.board[fc] != EMPTY) return false; // must be empty
    if(fc == pos.ko) return false;           // can't retake Ko immediately
    if(fc == NN) return true; // pass always legal

    
    
  
    // Make a copy to see if it results in suicide
    Position temp = pos;
    temp.board[fc] = color;
    // Capture opponent stones
    int opp_color = swap_color(color);

    // Need to see if we capture any neighbor groups of opposite color
    // or if the placed stone itself is captured (suicide).
    for(int i = 0; i < NEIGHBORS[fc].count; i++){
        int nb = NEIGHBORS[fc].neighbors[i];
        if(temp.board[nb] == opp_color) {
            maybe_capture_stones(temp, nb);
        }
    }


    // Also check if we are suiciding
    auto captured = maybe_capture_stones(temp, fc);
    if(!captured.empty()) {
        // It's suicide if we just captured ourselves
        return false;
    }
    // std::cout << "bbbb " << fc << '\n';
    return true;
}

// Execute a move, returning a new Position
// WE know that move is legal, so we don't need to check for suicide
Position play_move(const Position &oldPos, int fc) {
    Position pos = oldPos; 
    int color = pos.to_move;

    // Pass move
    if(fc == NN) {
        pos.to_move = swap_color(color);
        if(pos.pass_happened) {
            pos.is_game_over = true;
            return pos;
        }
        
        pos.pass_happened = true;
        return pos;
    }

  

    // Clear Ko
    pos.ko = -1;
    pos.board[fc] = color;
    pos.empty_spaces.erase(fc);  // Remove from empty spaces
    pos.pass_happened = false;

    int opp_color = swap_color(color);

    int maybe_ko_checker = -1;
    // Capture any opponent stones adjacent
    int total_opp_captured = 0;
    for(int i = 0; i < NEIGHBORS[fc].count; i++){
        int nb = NEIGHBORS[fc].neighbors[i];
        if(pos.board[nb] == opp_color) {
            auto captured = maybe_capture_stones(pos, nb);
            total_opp_captured += (int)captured.size();
            if((int)captured.size() == 1){
                maybe_ko_checker = nb;
            }
        }
    }


    // Check for Ko: if exactly 1 stone was captured and the new stone is in a 
    // one-point eye shape, set Ko
    if(total_opp_captured == 1 && is_koish_for_next_player(pos, maybe_ko_checker, fc)) {
        pos.ko = maybe_ko_checker; 
    }

    // Next player
    pos.to_move = opp_color;

    return pos;
}

// For final scoring: 
//   We fill in connected empties by whichever color encloses them exclusively.
//   If both or neither, fill as neutral.
double final_score(const Position &pos) {
    // Copy board so we can fill in territory
    Position temp = pos;
    // We'll do a naive approach: for each empty region, see which color(s) border it.
    // If purely black => fill black, purely white => fill white, else fill neutral.
    for(int i = 0; i < NN; i++){
        if(temp.board[i] == EMPTY) {
            auto [chain, reached] = find_reached(temp, i);
            // Suppose the first neighbor color is the candidate
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
                // fill chain with candidate
                for(int fc : chain) {
                    temp.board[fc] = candidate;
                }
            } else {
                // fill chain with '?' => treat as neutral
                for(int fc : chain) {
                    temp.board[fc] = -1;  // mark neutral
                }
            }
        }
    }
    // Now count
    int black_count = 0, white_count = 0;
    for(int i=0; i<NN; i++){
        if(temp.board[i] == BLACK) black_count++;
        else if(temp.board[i] == WHITE) white_count++;
    }
    // White gets komi
    double score = (double)black_count - (double)white_count + 0.0;
    score -= KOMI; 
    return score; 
}