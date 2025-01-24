#include "mcts.cuh"
#include <iostream>
#include <string>

int get_human_move(const Position &pos) {
    while(true) {
        std::cout << "Enter your move as 'row col' or 'pass': ";
        std::string input;
        std::cin >> input;
        if(!std::cin.good()) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            continue;
        }
        if(input == "pass") {
            return NN; // pass
        } else {
            try {
                int r = std::stoi(input);
                int c;
                std::cin >> c;
                if(r > 0 && r <= N && c > 0 && c <= N) {
                    // Flatten
                    return (r-1)*N + (c-1);
                } else {
                    std::cout << "Invalid row/col.\n";
                }
            } catch(...) {
                std::cout << "Invalid input.\n";
            }
        }
    }
    return NN;
}

int main() {
    std::cout << "Select mode:\n"
              << "1) Human vs AI\n"
              << "2) AI vs AI\n"
              << "Enter choice: ";
    int mode;
    std::cin >> mode;

    Position rootPos;
    std::unique_ptr<Node> root = std::make_unique<Node>(rootPos, nullptr, -1, 0, EMPTY);

    int black_iters=1, black_sims=1;
    int white_iters=1, white_sims=1;

    bool isHumanPlaying = false;
    int humanColor = EMPTY;

    if(mode == 1) {
        isHumanPlaying = true;
        std::cout << "Choose your color (b/w): ";
        char colorChoice; 
        std::cin >> colorChoice;
        if(colorChoice == 'b' || colorChoice == 'B') humanColor = BLACK;
        else                                         humanColor = WHITE;

        std::cout << "AI MCTS - number of iterations: ";
        std::cin >> black_iters;
        std::cout << "AI MCTS - number of simulations per iteration: ";
        std::cin >> black_sims;

        white_iters = black_iters;
        white_sims  = black_sims;
    }
    else if(mode == 2) {
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
        int toMove = root->state.to_move;
        bool currentPlayerIsHuman = (isHumanPlaying && toMove == humanColor);

        int fc;
        if(currentPlayerIsHuman) {
            fc = get_human_move(root->state);
        } else {
            int iters = (toMove == BLACK ? black_iters : white_iters);
            int sims  = (toMove == BLACK ? black_sims  : white_sims);

            for(int i=0; i < iters; i++) {
                mcts_iteration(root.get(), sims);
            }
            fc = best_move(root.get());
        }

        if(fc < 0) {
            std::cout << "No moves available. Game ends.\n";
            break;
        }

        auto rc = unflatten(fc);
        if(fc < NN) {
            std::cout << (toMove == BLACK ? "Black" : "White") 
                      << " plays (" << rc.first+1 << "," << rc.second+1 << ")\n";
        } else {
            std::cout << (toMove == BLACK ? "Black" : "White") 
                      << " plays (pass)\n";
        }
        Position newPos = play_move(root->state, fc);
        newPos.print();
        std::cout << "score so far = " << final_score(newPos) << "\n";

        root = std::make_unique<Node>(newPos, nullptr, -1, root->move_number+1, root->state.to_move);

        if(root->state.is_game_over) {
            std::cout << "Game is over!\n";
            break;
        }
    }

    double score = final_score(root->state);
    std::cout << "Final Score (Black - White - Komi) = " << score << "\n";
    if(score > 0) {
        std::cout << "BLACK wins by " << score << "!\n";
    } else if(score < 0) {
        std::cout << "WHITE wins by " << -score << "!\n";
    } else {
        std::cout << "It's a draw.\n";
    }

    return 0;
}
