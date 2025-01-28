// #include "neighbors.hpp"
// #include "position.hpp"
#include "mcts.cuh"

#include <iostream>
#include <string>

// Helper function to read a move from the human
// Returns the flattened coordinate or NN for pass
int get_human_move(const Position& pos) {

    if(pos.is_game_over == true) return -1;

    while (true) {
        std::cout << "Enter your move as 'row col' or 'pass': ";
        std::string input;
        std::cin >> input;
        if (!std::cin.good()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            continue;
        }

        if (input == "pass") {
            return NN; // pass move
        } else {
            // Attempt to parse row and column
            try {
                int r = std::stoi(input);
                int c;
                std::cin >> c;  // read second number
                // Basic range check (assuming board is NxN)

                if (r > 0 && r <= N && c > 0 && c <= N) {

                    int move = flatten(r - 1, c - 1);

                    if (move == pos.ko) {
                        std::cout << "Invalid row/col. KO!\n";
                        continue;
                    }

                    if (pos.board[move] != EMPTY) {
                        std::cout << "Invalid row/col. Choose Empty cell\n";
                        continue;
                    }

                    return move;
                } else {
                    std::cout << "Invalid row/col. Please try again.\n";
                }
            } catch (...) {
                std::cout << "Invalid input. Please try again.\n";
            }
        }
    }
    // Fallback
    return NN;
}

int main() {
    // Prompt for mode
    


    Array4Neighbors H_neighbors[NN];
    build_neighbors_array(H_neighbors);

    std::cout << "Select mode:\n";
    std::cout << "1) Human vs AI\n";
    std::cout << "2) AI vs AI\n";
    std::cout << "Enter choice: ";
    int mode;
    std::cin >> mode;


    // Common game initialization
    Position rootPos;


    Node root = Node(rootPos, nullptr, -1, 0, EMPTY, 0, H_neighbors);


    // Variables for MCTS parameters
    int black_iters = 1, black_sims = 1;
    int white_iters = 1, white_sims = 1;

    // Variable to indicate if there's a human and what color they play
    bool isHumanPlaying = false;
    int humanColor = EMPTY; // Will be set to BLACK or WHITE if in human vs AI

    // Mode 1: Human vs AI
    if (mode == 1) {
        isHumanPlaying = true;

        // Ask which color the human will play
        std::cout << "Choose your color (b/w): ";
        char colorChoice;
        std::cin >> colorChoice;
        if (colorChoice == 'b' || colorChoice == 'B') {
            humanColor = BLACK;
        } else {
            humanColor = WHITE;
        }

        // For the AI, ask for MCTS parameters
        std::cout << "AI MCTS - Number of iterations: ";
        std::cin >> black_iters;  // We'll use black_iters/black_sims for whichever side is AI
        std::cout << "AI MCTS - Number of simulations per iteration: ";
        std::cin >> black_sims;

        // We'll store the same parameters in black_iters/black_sims and white_iters/white_sims
        // so that the AI uses the same iteration/simulation count, no matter which color.
        white_iters = black_iters;
        white_sims  = black_sims;



    } 
    // Mode 2: AI vs AI
    else if (mode == 2) {
        std::cout << "For Black AI:\n";
        std::cout << "  MCTS Iterations: ";
        std::cin >> black_iters;
        std::cout << "  MCTS Simulations: ";
        std::cin >> black_sims;

        std::cout << "For White AI:\n";
        std::cout << "  MCTS Iterations: ";
        std::cin >> white_iters;
        std::cout << "  MCTS Simulations: ";
        std::cin >> white_sims;
    }
    else {
        std::cout << "Invalid mode. Exiting.\n";
        return 1;
    }

    // Main game loop
    for(int moveNumber = 1; moveNumber <= 201; moveNumber++) {

        std::cout << "=============== Move #" << moveNumber << " ===============\n";
        int toMove = root.state.to_move;



        // Determine if current player is human or AI
        bool currentPlayerIsHuman = (isHumanPlaying && toMove == humanColor);

        int fc;
        int best_child_id;
        if (currentPlayerIsHuman) {
            // Human move
            fc = get_human_move(root.state);


        } else {
            // AI move
            int iters = (toMove == BLACK ? black_iters : white_iters);
            int n_simulations = (toMove == BLACK ? black_sims : white_sims);

            for(int i = 0; i < iters; i++) {
                mcts_iteration(&root, n_simulations);
          
            }
            best_child_id = find_best_child(&root);
            fc = root.children[best_child_id]->move_fc;


            double c_w = double(root.children[best_child_id]->wins);
            double c_v = double(root.children[best_child_id]->visits);
            std::cout << "ratio " << (c_w / c_v) << '\n';
        }

        if(fc < 0) {
            // No moves available
            std::cout << "No moves available. Game ends.\n";
            break;
        }

        
        // Display chosen move
        auto rc = unflatten(fc);
        std::cout << (toMove == BLACK ? "Black" : "White");
        if (fc < NN) {
            std::cout << " plays (" << rc.first + 1 << "," << rc.second + 1<< ")\n";
        } else {
            std::cout << " plays (pass)\n";
        }

        // Apply the move
        Position newPos = play_move(root.state, fc, H_neighbors);
        newPos.print();
        std::cout << "score " << final_score(newPos, H_neighbors) << "!\n";

        int new_move_number = root.move_number + 1;
        int node_color = root.state.to_move;


        // Create new root node
        root = Node(newPos, nullptr, -1, new_move_number, node_color, 0, H_neighbors);
    }

    // End: compute final score
    double score = final_score(root.state, H_neighbors);


    std::cout << "Final Score (Black - White - Komi): " << score << "\n";
    if(score > 0) {
        std::cout << "BLACK wins by " << score << "!\n";
    } else if (score < 0) {
        std::cout << "WHITE wins by " << -score << "!\n";
    } else {
        std::cout << "It's a draw!\n";
    }

    return 0;
}
