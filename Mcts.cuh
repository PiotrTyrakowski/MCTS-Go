////////////////////////////////////////////////////////////////////////////////
// MCTS Node structure (CPU side)
////////////////////////////////////////////////////////////////////////////////
struct MCTSNode {
    std::vector<unsigned char> board;   // Current board state at this node
    unsigned char to_move;             // Whose turn is it in this node
    int visits;
    double wins;  // for the player to_move (or you can store separately)
    
    std::vector<std::unique_ptr<MCTSNode>> children;
    std::vector<int> valid_moves; // store the moves that create these children

    MCTSNode(const std::vector<unsigned char> &b, unsigned char tm)
        : board(b), to_move(tm), visits(0), wins(0.0)
    {}

    // Expand child nodes for each valid move
    void expand() {
        // gather empty points
        valid_moves.clear();
        for (int i = 0; i < NN; i++) {
            if (board[i] == EMPTY) {
                valid_moves.push_back(i);
            }
        }
        // create child boards
        for (auto mv : valid_moves) {
            std::vector<unsigned char> child_board = board;
            play_move_incomplete_cpu(child_board, mv, to_move);
            std::unique_ptr<MCTSNode> child(
                new MCTSNode(child_board, swap_colors(to_move))
            );
            children.push_back(std::move(child));
        }
    }

    bool is_leaf() const {
        return children.empty();
    }
};

////////////////////////////////////////////////////////////////////////////////
// MCTS: selection function (simplified UCB1)
////////////////////////////////////////////////////////////////////////////////
static inline double ucb_value(int parent_visits, double child_wins, int child_visits, double c=1.4) {
    if (child_visits == 0) {
        return 1e9; // effectively infinite
    }
    return (child_wins / child_visits) + c * sqrt(log((double)parent_visits) / child_visits);
}

MCTSNode* select_child_uct(MCTSNode* node) {
    MCTSNode *best = nullptr;
    double best_value = -1e30;
    for (size_t i = 0; i < node->children.size(); i++) {
        MCTSNode *c = node->children[i].get();
        double val = ucb_value(node->visits, c->wins, c->visits);
        if (val > best_value) {
            best_value = val;
            best = c;
        }
    }
    return best;
}