#ifndef POSITION_CUH
#define POSITION_CUH

#include "neighbors.cuh"
#include <array>
#include <unordered_set>
#include <vector>
#include <utility>

////////////////////////////////////////////////////////////
// We declare everything that might be needed from device or host
////////////////////////////////////////////////////////////

HD inline int swap_color(int color) {
    if(color == BLACK) return WHITE;
    if(color == WHITE) return BLACK;
    return color;
}

// Forward declarations of structures/functions
struct Position;

// Because BFS-based capturing uses STL containers not supported on device,
// we will mark them as host-only in the .cu file. The function signatures
// remain normal here (they're eventually implemented in position.cu).
void bulk_remove_stones(Position &pos, const std::vector<int> &stones);

std::pair<std::vector<int>, std::vector<int>> find_reached(const Position &pos, int start);

std::vector<int> maybe_capture_stones(Position &pos, int fc);

bool is_koish_for_next_player(const Position &pos, int maybe_ko_checker, int played_stone);

bool is_legal_move(const Position &pos, int fc, int color);

Position play_move(const Position &oldPos, int fc);

double final_score(const Position &pos);

struct Position {
    std::array<int, NN> board;
    int ko;
    int to_move;
    bool pass_happened;
    bool is_game_over;

  
    std::unordered_set<int> empty_spaces;

    Position();
    Position(const std::array<int, NN>& board_, int ko_, int to_move_, 
             bool pass_happened_, bool is_game_over_);

    void print() const;
};

#endif // POSITION_CUH
