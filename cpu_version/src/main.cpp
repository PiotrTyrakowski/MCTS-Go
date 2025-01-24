// #include "neighbors.hpp"
// #include "position.hpp"
#include "mcts.hpp"

#include <iostream>
#include <vector>
#include <array>

int main() {

  
    Position rootPos;
    std::unique_ptr<Node> root = std::make_unique<Node>(rootPos, nullptr, -1, 0, EMPTY);


    for(int moveNumber = 1; moveNumber <= 201; moveNumber++) {
        std::cout << "Running MCTS iteration for move #" << moveNumber << "...\n"; // Debug output

        // Run MCTS
        int iters = 5;
        int n_simulations = 300;
        
        // MCTS_SIMULATIONS
        for(int i=0; i<iters; i++){
            mcts_iteration(root.get(), n_simulations);
        }

        std::cout << root.get()->color_of_move <<" wins  " << root.get()->wins << '\n';
        std::cout << root.get()->color_of_move << " plays " << root.get()->visits << '\n';

       
        std::cout << "MCTS iterations completed for move #" << moveNumber << "\n"; // Debug output

        // Pick best move
        int fc = best_move(root.get());

        if(fc < 0) {
            std::cout << "No moves available. Game ends.\n";
            break;
        }

        if(fc == NN) {
            for(const auto& child_ptr : root->children) {
                std::cout <<"pas  wins  " << child_ptr->wins << "\n";
                std::cout <<"pas  plays " << child_ptr->visits << "\n";
                std::cout <<"pas  move  " << child_ptr->move_fc << "\n\n";

            }

        }
        

        // Display chosen move
        auto rc = unflatten(fc);
        if (fc < NN)
        {
            std::cout << "Move #" << moveNumber << " for "
                    << (root->state.to_move == BLACK ? "Black" : "White")
                    << " => (" << rc.first << "," << rc.second << ")\n";
        }
        else
        {
            std::cout << "Move #" << moveNumber << " for "
                    << (root->state.to_move == BLACK ? "Black" : "White")
                    << " => (pass)\n";
        }

        // Apply the move
        Position newPos = play_move(root->state, fc);
        // Print board
        newPos.print();

        // Create new root node
        root = std::make_unique<Node>(newPos, nullptr, -1, root->move_number + 1, root->state.to_move);
    }

    // End: compute final score
    double score = final_score(root->state);
    std::cout << "Final Score (Black - White - Komi): " << score << "\n";
    if(score > 0) {
        std::cout << "BLACK wins by " << score << "!\n";
    } else {
        std::cout << "WHITE wins by " << -score << "!\n";
    }

    return 0;
}
