#include "position.cuh"
#include "constants.cuh"



// HOSTDEV ArrayInt NEIGHBORS[NN] = build_neighbors_array();

// ArrayInt h_neighbors_array[NN];
// build_neighbors_array(h_neighbors_array);



// Helper: Bulk remove stones from board
HOSTDEV void bulk_remove_stones(Position &pos, const ArrayInt &stones) {
    for(int i = 0; i < stones.size(); i++) {
        int fc = stones[i];
        pos.board[fc] = EMPTY;
        pos.empty_spaces.insert(fc);
    }
}

// Finds two things:
// 1. A chain: all connected stones of the same color starting from 'start'
// 2. All points adjacent to this chain (reached points)
// Returns: pair of vectors {chain, reached}
HOSTDEV ArrayIntPair find_reached(const Position &pos, int start, Array4Neighbors* neighbors_array)
{
    // Get the color of the starting point (BLACK, WHITE, or EMPTY)
    int color = pos.board[start];
    
    // Will store all connected stones of the same color
    ArrayInt chain;
    
    // Will store all points adjacent to the chain
    ArrayInt reached;

    // Keep track of which points we've already processed
    bool visited[NN];
    for(int i = 0; i<NN; i++) 
        visited[i] = false;
        
    visited[start] = true;
    
    // BFS queue for exploring connected stones
    Queue frontier;
    frontier.push(start);
    chain.push_back(start);

    // Breadth-first search through connected stones
    while(!frontier.empty()){
        // get first element and remove it
        int current = frontier.pop();
        
        // Check all neighboring points
        for(int i = 0; i < neighbors_array[current].size(); i++){
            int nb = neighbors_array[current].array[i];
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
    
    return ArrayIntPair(chain, reached);
}

// Attempt to capture a chain if it has no liberties
// Returns the set of captured stones (if any).
HOSTDEV ArrayInt maybe_capture_stones(Position &pos, int fc, Array4Neighbors* neighbors_array) {
    // Find all connected stones of the same color and their adjacent points
    ArrayIntPair chain_and_reached = find_reached(pos, fc, neighbors_array);
    ArrayInt chain = chain_and_reached.first;
    ArrayInt reached = chain_and_reached.second;


    // return empty if whole board is one color because there is no capture
    if (chain.size() == NN) return ArrayInt();

    // Check if the chain has any liberties (empty adjacent points)
    bool has_liberty = false;
    for(int i = 0; i < reached.size(); i++) {
        int r = reached[i];
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
    return ArrayInt();  // Return empty vector if no stones were captured
}

// Check if fc is "ko-ish": the move just captured exactly 1 stone
// and left a single surrounded point with no liberties.
HOSTDEV bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone, Array4Neighbors* neighbors_array) {

    // Check if the played stone is empty
    if(pos.board[played_stone] == EMPTY) return false;

    // Check if the maybe_ko_checker is empty
    if(pos.board[maybe_ko_checker] != EMPTY) return false;


    int played_stone_color = pos.board[played_stone];

   
    for(int i = 0; i < neighbors_array[played_stone].size(); i++) {
        int nb = neighbors_array[played_stone].array[i];

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
HOSTDEV bool is_legal_move(const Position &pos, int fc, int color, Array4Neighbors* neighbors_array) {
    if(pos.is_game_over) return false;
    if(fc < 0 || fc > NN) return false;
    if(fc == NN) return true; // pass always legal
    if(pos.board[fc] != EMPTY) return false; // must be empty
    if(fc == pos.ko) return false;           // can't retake Ko immediately

    
    
  
    // Make a copy to see if it results in suicide
    Position temp = pos;
    temp.board[fc] = color;
    // Capture opponent stones
    int opp_color = swap_color(color);

    // Need to see if we capture any neighbor groups of opposite color
    // or if the placed stone itself is captured (suicide).
    for(int i = 0; i < neighbors_array[fc].size(); i++){
        int nb = neighbors_array[fc].array[i];
        if(temp.board[nb] == opp_color) {
            maybe_capture_stones(temp, nb, neighbors_array);
        }
    }


    // Also check if we are suiciding
    auto captured = maybe_capture_stones(temp, fc, neighbors_array);
    if(captured.size() > 0) {
        // It's suicide if we just captured ourselves
        return false;
    }
    return true;
}

// Execute a move, returning a new Position
// WE know that move is legal, so we don't need to check for suicide
HOSTDEV Position play_move(const Position &oldPos, int fc, Array4Neighbors* neighbors_array) {
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
    pos.empty_spaces.remove(fc);  // Remove from empty spaces
    pos.pass_happened = false;

    int opp_color = swap_color(color);

    int maybe_ko_checker = -1;
    // Capture any opponent stones adjacent
    int total_opp_captured = 0;
    for(int i = 0; i < neighbors_array[fc].size(); i++){
        int nb = neighbors_array[fc].array[i];
        if(pos.board[nb] == opp_color) {
            ArrayInt captured = maybe_capture_stones(pos, nb, neighbors_array);
            total_opp_captured += captured.size();
            if(captured.size() == 1){
                maybe_ko_checker = nb;
            }
        }
    }


    // Check for Ko: if exactly 1 stone was captured and the new stone is in a 
    // one-point eye shape, set Ko
    if(total_opp_captured == 1 && is_koish_for_next_player(pos, maybe_ko_checker, fc, neighbors_array)) {
        pos.ko = maybe_ko_checker; 
    }

    // Next player
    pos.to_move = opp_color;

    return pos;
}

// For final scoring: 
//   We fill in connected empties by whichever color encloses them exclusively.
//   If both or neither, fill as neutral.
HOSTDEV double final_score(const Position &pos, Array4Neighbors* neighbors_array) {
    // Copy board so we can fill in territory
    Position temp = pos;
    // We'll do a naive approach: for each empty region, see which color(s) border it.
    // If purely black => fill black, purely white => fill white, else fill neutral.
    for(int i = 0; i < NN; i++){
        if(temp.board[i] == EMPTY) {
            auto [chain, reached] = find_reached(temp, i, neighbors_array);
            // Suppose the first neighbor color is the candidate
            int candidate = -1;
            bool mixed = false;
            for(int i = 0; i < reached.size(); i ++) 
            {
                int r = reached.array[i];
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
                for(int i = 0; i < chain.size(); i++) 
                {
                    int fc = chain.array[i];
                    temp.board[fc] = candidate;
                }
            } else {
                for(int i = 0; i < chain.size(); i++) 
                {
                    // fill chain with '?' => treat as neutral
                    int fc = chain.array[i];
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