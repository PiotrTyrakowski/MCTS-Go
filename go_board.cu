/**
 * go_board.cu
 *
 * A minimal CUDA/C++ implementation of Go board logic:
 *
 *  - Storing the board
 *  - Scoring
 *  - Getting empty spaces
 *  - Playing moves with capture & suicide checks
 *
 * NOTE:
 *  1) This is demonstration-level code. In a full engine, you may want
 *     to refine data structures or parallelize certain calls using
 *     CUDA kernels, especially for large-scale Monte Carlo rollouts.
 *  2) This code compiles with nvcc and runs primarily on host side
 *     (with some __host__ __device__ inlines for convenience). For
 *     multi-thread or GPU parallelization, you'd add kernels.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>  // std::find, std::count
#include <queue>      // for BFS frontier

// ---------------------------------------------------
// Constants & Utility Macros
// ---------------------------------------------------

static const int N       = 19;            // Board dimension
static const int NN      = N * N;         // 19x19
static const char WHITE  = 'O';           // White stone
static const char BLACK  = 'X';           // Black stone
static const char EMPTY  = '.';           // Empty intersection
static const char DEAD   = '?';           // For scoring (neutral territory)

// Helper to swap black & white
__host__ __device__
inline char swapColor(const char c) {
    if (c == BLACK) return WHITE;
    if (c == WHITE) return BLACK;
    return c; // else remain the same
}

// Flatten (row,col)->fc
__host__ __device__
inline int flatten(int row, int col) {
    return row * N + col;
}

// Unflatten (fc)->(row, col)
__host__ __device__
inline void unflatten(int fc, int &r, int &c) {
    r = fc / N;
    c = fc % N;
}

// Check if coords are on board
__host__ __device__
inline bool isOnBoard(int row, int col) {
    return (row >= 0 && row < N && col >= 0 && col < N);
}

// ---------------------------------------------------
// A small device/host helper to get neighbors
// ---------------------------------------------------
__host__ __device__
inline thrust::device_vector<int> getNeighbors(int fc) {
    thrust::device_vector<int> result;
    result.reserve(4);

    int r, c;
    unflatten(fc, r, c);

    // Up
    if (isOnBoard(r - 1, c)) thrust::copy(&flatten(r - 1, c), &flatten(r - 1, c) + 1, thrust::back_inserter(result));
    // Down 
    if (isOnBoard(r + 1, c)) thrust::copy(&flatten(r + 1, c), &flatten(r + 1, c) + 1, thrust::back_inserter(result));
    // Left
    if (isOnBoard(r, c - 1)) thrust::copy(&flatten(r, c - 1), &flatten(r, c - 1) + 1, thrust::back_inserter(result));
    // Right
    if (isOnBoard(r, c + 1)) thrust::copy(&flatten(r, c + 1), &flatten(r, c + 1) + 1, thrust::back_inserter(result));

    return result;
}

// ---------------------------------------------------
// GoBoard class
// ---------------------------------------------------
class GoBoard {
private:
    // The board array. Each cell can be 'X', 'O', or '.'
    // If you want to run thousands of rollouts on GPU, consider
    // storing this in device memory as well, or use pinned memory.
    char board[NN];

    // We store Ko if needed. For simplicity, store as an index
    // or use -1 to indicate "no Ko".
    int koIndex;

    // -------------------------------------------
    // Helper: BFS to find group chain & reached
    //
    //  chain[]  -> all stones in the group
    //  reached[]-> neighbors of that chain
    // -------------------------------------------
    void findReached(int fc,
                     std::vector<bool> &chain,
                     std::vector<bool> &reached) const
    {
        char color = board[fc];
        std::queue<int> frontier;
        frontier.push(fc);
        chain[fc] = true;

        while (!frontier.empty()) {
            int current = frontier.front();
            frontier.pop();

            auto nbrs = getNeighbors(current);
            for (int nb : nbrs) {
                if (board[nb] == color && !chain[nb]) {
                    chain[nb] = true;
                    frontier.push(nb);
                }
                else if (board[nb] != color) {
                    reached[nb] = true;
                }
            }
        }
    }

    // -------------------------------------------
    // Helper: capture chain if it has no liberties
    // Returns the stones that were captured (if any)
    // -------------------------------------------
    std::vector<int> maybeCaptureStones(int fc) {
        std::vector<bool> chain(NN, false);
        std::vector<bool> reached(NN, false);

        findReached(fc, chain, reached);

        // Check if chain is fully surrounded (no empty in reached)
        bool hasLiberty = false;
        for (int i = 0; i < NN; i++) {
            if (reached[i] && board[i] == EMPTY) {
                hasLiberty = true;
                break;
            }
        }
        if (!hasLiberty) {
            // Capture
            std::vector<int> capturedStones;
            for (int i = 0; i < NN; i++) {
                if (chain[i]) {
                    board[i] = EMPTY;
                    capturedStones.push_back(i);
                }
            }
            return capturedStones;
        }
        return {};
    }

    // -------------------------------------------
    // Helper: check if a space is "koish"
    //   i.e. if empty spot fc is surrounded by exactly
    //   one color. If so, return that color. Otherwise, 0.
    // -------------------------------------------
    char isKoish(int fc) const {
        if (board[fc] != EMPTY) return 0;
        auto nbrs = getNeighbors(fc);
        if (nbrs.empty()) return 0;

        char firstColor = 0;
        for (int nb : nbrs) {
            if (board[nb] != EMPTY) {
                if (firstColor == 0) {
                    firstColor = board[nb];
                } else if (board[nb] != firstColor) {
                    // more than one color
                    return 0;
                }
            }
        }
        // If we have a single color neighbor
        // but no empty neighbor, it might be Ko
        // Return that color
        if (firstColor != 0) {
            // Double-check that there are no empty neighbors
            for (int nb : nbrs) {
                if (board[nb] == EMPTY) {
                    return 0;
                }
            }
            return firstColor;
        }
        return 0;
    }

    // -------------------------------------------
    // Helper: count stones of each color
    //  purely for final scoring
    // -------------------------------------------
    void fillTerritoriesForScoring(char territoryColor) {
        for (int i = 0; i < NN; i++) {
            if (board[i] == EMPTY) {
                // BFS from this empty intersection
                std::vector<bool> chain(NN, false);
                std::vector<bool> reached(NN, false);
                findReached(i, chain, reached);

                // If reached encloses exactly one color, fill with that color
                char possible = 0;
                for (int r = 0; r < NN; r++) {
                    if (reached[r] && board[r] != EMPTY && board[r] != DEAD) {
                        if (possible == 0) {
                            possible = board[r];
                        } else if (board[r] != possible) {
                            possible = 0; // multiple colors => neutral
                            break;
                        }
                    }
                }
                char fillChar = (possible == 0) ? DEAD : possible;
                // Mark chain with fillChar
                for (int c = 0; c < NN; c++) {
                    if (chain[c]) {
                        board[c] = fillChar;
                    }
                }
            }
        }
    }

public:
    // -------------------------------------------
    // Constructor
    // -------------------------------------------
    GoBoard() {
        // Initialize empty board
        for (int i = 0; i < NN; i++) {
            board[i] = EMPTY;
        }
        koIndex = -1;
    }

    // -------------------------------------------
    // Accessors
    // -------------------------------------------
    __host__ __device__
    char getStone(int fc) const {
        return board[fc];
    }

    __host__ __device__
    void setStone(int fc, char c) {
        board[fc] = c;
    }

    // For debugging or printing
    void printBoard() const {
        for (int r = 0; r < N; r++) {
            for (int c = 0; c < N; c++) {
                std::cout << board[flatten(r,c)];
            }
            std::cout << "\n";
        }
    }

    // -------------------------------------------
    // Return a list of empty spaces
    //   (for candidate moves in Monte Carlo)
    // -------------------------------------------
    std::vector<int> getEmptySpaces() const {
        std::vector<int> empties;
        empties.reserve(NN);
        for (int i = 0; i < NN; i++) {
            if (board[i] == EMPTY) {
                empties.push_back(i);
            }
        }
        return empties;
    }

    // -------------------------------------------
    // Score the board
    //  Returns (black_count - white_count)
    // -------------------------------------------
    int scoreBoard() {
        // Copy the board to avoid mutating the real board
        char backup[NN];
        for (int i = 0; i < NN; i++) {
            backup[i] = board[i];
        }

        // Fill territories
        fillTerritoriesForScoring(DEAD);

        int blackCount = 0;
        int whiteCount = 0;
        for (int i = 0; i < NN; i++) {
            if (board[i] == BLACK) blackCount++;
            if (board[i] == WHITE) whiteCount++;
        }

        // Restore the board
        for (int i = 0; i < NN; i++) {
            board[i] = backup[i];
        }

        return blackCount - whiteCount;
    }

    // -------------------------------------------
    // Attempt to play a move at flat coordinate fc
    // Returns true if move is valid, false otherwise
    // -------------------------------------------
    bool playMove(int fc, char color) {
        // Check Ko
        if (fc == koIndex) {
            // Ko violation
            return false;
        }

        // Check if empty
        if (board[fc] != EMPTY) {
            // Not empty => illegal
            return false;
        }

        // Place stone tentatively
        board[fc] = color;

        // Capture opponent neighbors
        char oppColor = swapColor(color);
        std::vector<int> oppStones;
        auto nbrs = getNeighbors(fc);
        for (int nb : nbrs) {
            if (board[nb] == oppColor) {
                oppStones.push_back(nb);
            }
        }

        int oppCapturedCount = 0;
        for (int fs : oppStones) {
            // maybeCaptureStones will only capture if group has no liberties
            auto captured = maybeCaptureStones(fs);
            oppCapturedCount += static_cast<int>(captured.size());
        }

        // Check for suicide
        // The newly placed stone might have no liberties if it's not capturing anything.
        // So we test if it got captured by the same logic:
        auto myCaptured = maybeCaptureStones(fc);
        if (!myCaptured.empty()) {
            // This means playing here is suicidal.
            // Revert move
            board[fc] = EMPTY;
            return false;
        }

        // Ko check:
        //  If exactly 1 stone was captured and the new move is "koish",
        //  mark fc as the Ko index for next turn
        koIndex = -1;
        if (oppCapturedCount == 1) {
            char isKoColor = isKoish(fc);
            if (isKoColor == oppColor) {
                koIndex = fc;  // set Ko
            }
        }

        return true;
    }
};

// ---------------------------------------------------
// Example of how you might test/use the class
// ---------------------------------------------------
int main() {
    GoBoard board;
    std::cout << "Empty board:\n";
    board.printBoard();
    std::cout << "\n";

    // Let's try a couple of moves
    // Flattened coords: (r=3, c=3) -> 3*N+3 = 3*19+3=60
    int moveFC = 60;
    bool success = board.playMove(moveFC, BLACK);
    std::cout << "Play black at (3,3), success=" << success << "\n";
    board.printBoard();
    std::cout << "\n";

    // Score
    int sc = board.scoreBoard();
    std::cout << "Current score (black - white) = " << sc << "\n";

    // Get empty spaces
    auto empties = board.getEmptySpaces();
    std::cout << "Number of empty points: " << empties.size() << "\n";

    return 0;
}
