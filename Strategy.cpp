#include <iostream>
#include <unistd.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <algorithm>
#include "Point.h"
#include "Strategy.h"
#include "Judge.h"

using namespace std;

// --- Debugging Switch ---
// Comment out the following line when submitting to avoid any output overhead
// #define DEBUG_MODE

#ifdef DEBUG_MODE
#define DEBUG_LOG(x) do { std::cerr << x << std::endl; } while (0)
#else
#define DEBUG_LOG(x) do {} while (0)
#endif

// --- Constants ---
const int MAX_M = 12;
const int MAX_N = 12;
const float C_PUCT = 1.0f; // PUCT exploration constant

// --- Neural Network Skeleton ---

// Assume offline trained weights are stored here.
// For the skeleton, we define dummy sizes.
const int INPUT_CHANNELS = 3; // e.g., current player, opponent, invalid spots
const int HIDDEN_DIM = 64;

// Dummy weights structure
struct NNWeights {
    // These would typically be statically loaded from a header file
    // Here we just declare what we would use
    std::vector<float> fc_policy_w; // Shape: [12 * 12 * INPUT_CHANNELS, MAX_N]
    std::vector<float> fc_value_w;  // Shape: [12 * 12 * INPUT_CHANNELS, 1]

    NNWeights() {
        // Initialize with dummy values (zeros/random) if empty
        fc_policy_w.assign(12 * 12 * INPUT_CHANNELS * MAX_N, 0.01f);
        fc_value_w.assign(12 * 12 * INPUT_CHANNELS * 1, 0.0f);
    }
};

static NNWeights g_weights;

// Simple structure to hold network output
struct NetworkOutput {
    std::vector<float> policy; // probabilities for each column (size N)
    float value;               // in range [-1, 1]
};

// Extremely simplified Forward Pass
// A real version would use dense matrix multiplication or small convolutions
NetworkOutput forwardPass(const std::vector<float>& input_tensor, int M, int N) {
    NetworkOutput out;
    out.policy.assign(N, 0.0f);

    // Minimal linear layer for policy
    std::vector<float> logits(N, 0.0f);
    float max_logit = -1e9f;
    for (int j = 0; j < N; ++j) {
        for (size_t i = 0; i < input_tensor.size(); ++i) {
            logits[j] += input_tensor[i] * g_weights.fc_policy_w[i * MAX_N + j];
        }
        if (logits[j] > max_logit) max_logit = logits[j];
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
        // Softmax components (subtract max_logit to avoid overflow)
        out.policy[j] = std::exp(logits[j] - max_logit);
        sum_exp += out.policy[j];
    }

    // Normalize policy (Softmax)
    for (int j = 0; j < N; ++j) {
        if (sum_exp > 0) out.policy[j] /= sum_exp;
        else out.policy[j] = 1.0f / N;
    }

    // Minimal linear layer for value
    float val = 0.0f;
    for (size_t i = 0; i < input_tensor.size(); ++i) {
        val += input_tensor[i] * g_weights.fc_value_w[i];
    }
    // Tanh activation for value
    out.value = std::tanh(val);

    return out;
}

// State Encoder: pads variable sized board into a fixed tensor
// Channels: 0 = Current Player, 1 = Opponent, 2 = Valid/Invalid (including bounds and noX/noY)
std::vector<float> encodeBoard(int M, int N, const std::vector<int>& board_1d, int currentPlayer, int noX, int noY) {
    std::vector<float> tensor(12 * 12 * INPUT_CHANNELS, 0.0f);
    int opponent = (currentPlayer == 1) ? 2 : 1;

    for (int r = 0; r < 12; ++r) {
        for (int c = 0; c < 12; ++c) {
            int idx_base = (r * 12 + c) * INPUT_CHANNELS;

            if (r < M && c < N) {
                // Inside actual board
                if (r == noX && c == noY) {
                    tensor[idx_base + 2] = 1.0f; // Invalid spot
                } else {
                    int piece = board_1d[r * N + c];
                    if (piece == currentPlayer) {
                        tensor[idx_base + 0] = 1.0f;
                    } else if (piece == opponent) {
                        tensor[idx_base + 1] = 1.0f;
                    }
                    // tensor[idx_base + 2] remains 0 (valid)
                }
            } else {
                // Outside actual board boundaries (Padding area)
                tensor[idx_base + 2] = 1.0f; // Invalid spot
            }
        }
    }
    return tensor;
}

// --- Game State Definition ---
struct GameState {
    int M, N;
    std::vector<int> board; // 1D representation
    std::vector<int> top;
    int currentPlayer; // 1 for user, 2 for program. So if it's program's turn, it's 2.
    int noX, noY;
    int lastMoveX, lastMoveY;

    GameState(int m, int n, const int* b, const int* t, int curr, int nx, int ny, int lx = -1, int ly = -1)
        : M(m), N(n), currentPlayer(curr), noX(nx), noY(ny), lastMoveX(lx), lastMoveY(ly) {
        board.assign(b, b + M * N);
        top.assign(t, t + N);
    }

    GameState(const GameState& other) = default;

    // Check if terminal state
    // Return: 1 if user wins, 2 if program wins, 0 if draw, -1 if not terminal
    int checkTerminal() const {
        if (lastMoveX == -1 || lastMoveY == -1) {
            // Need to scan entire board if no last move is provided (e.g., initial state)
            int* b2d[MAX_M];
            for (int i = 0; i < M; i++) {
                b2d[i] = const_cast<int*>(&board[i * N]);
            }

            bool uWin = false;
            bool pWin = false;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (b2d[i][j] == 1) {
                        if (userWin(i, j, M, N, b2d)) uWin = true;
                    } else if (b2d[i][j] == 2) {
                        if (machineWin(i, j, M, N, b2d)) pWin = true;
                    }
                }
            }

            bool tie = isTie(N, top.data());

            if (uWin) return 1;
            if (pWin) return 2;
            if (tie) return 0;
            return -1;
        } else {
            // We only need to check the vicinity of the last played move
            int* b2d[MAX_M];
            for (int i = 0; i < M; i++) {
                b2d[i] = const_cast<int*>(&board[i * N]);
            }

            // The last piece played was by the PREVIOUS player
            int prevPlayer = (currentPlayer == 1) ? 2 : 1;

            if (prevPlayer == 1 && userWin(lastMoveX, lastMoveY, M, N, b2d)) {
                return 1;
            } else if (prevPlayer == 2 && machineWin(lastMoveX, lastMoveY, M, N, b2d)) {
                return 2;
            }

            if (isTie(N, top.data())) {
                return 0;
            }

            return -1;
        }
    }

    // Play a move in column c, return true if successful
    bool playMove(int c) {
        if (c < 0 || c >= N || top[c] <= 0) return false;

        int r = top[c] - 1;
        board[r * N + c] = currentPlayer;
        top[c]--;

        lastMoveX = r;
        lastMoveY = c;

        // If the newly updated top points to the invalid position, decrement again
        if (top[c] > 0 && (top[c] - 1 == noX && c == noY)) {
            top[c]--;
        }

        currentPlayer = (currentPlayer == 1) ? 2 : 1;
        return true;
    }
};

// --- MCTS Node ---
struct MCTSNode {
    int move; // column index that led to this node
    int currentPlayer; // player who is about to play from this node
    int N_visits;
    float W_total_value;
    float Q_mean_value;
    float P_prior;

    MCTSNode* parent;
    std::vector<MCTSNode*> children;

    // Store valid moves to expand
    std::vector<int> unexpanded_moves;
    bool is_terminal;
    int terminal_value; // 1 for user win, 2 for program win, 0 for draw

    MCTSNode(int m, int player, float prior, MCTSNode* p, GameState& state)
        : move(m), currentPlayer(player), N_visits(0), W_total_value(0.0f),
          Q_mean_value(0.0f), P_prior(prior), parent(p), is_terminal(false), terminal_value(-1) {

        int term = state.checkTerminal();
        if (term != -1) {
            is_terminal = true;
            terminal_value = term;
        } else {
            for (int c = 0; c < state.N; ++c) {
                if (state.top[c] > 0) {
                    unexpanded_moves.push_back(c);
                }
            }
        }
    }

    ~MCTSNode() {
        for (MCTSNode* child : children) {
            delete child;
        }
    }

    bool isFullyExpanded() const {
        return unexpanded_moves.empty() || is_terminal;
    }

    MCTSNode* getBestUCTChild() {
        MCTSNode* best_child = nullptr;
        float best_uct = -1e9f;

        for (MCTSNode* child : children) {
            float uct;
            if (child->N_visits == 0) {
                uct = C_PUCT * child->P_prior * std::sqrt((float)this->N_visits + 1e-8f); // Encourage exploration
            } else {
                // Perspective: Q is stored as value for the node's current player.
                // When picking a child, we want the highest value for THIS node's player.
                // Value stored in child->Q_mean_value is from the perspective of child->currentPlayer.
                // Since this is a zero-sum game, value for this player = -value for child player.
                float q_val = -child->Q_mean_value;
                float u_val = C_PUCT * child->P_prior * std::sqrt((float)this->N_visits) / (1.0f + child->N_visits);
                uct = q_val + u_val;
            }

            if (uct > best_uct) {
                best_uct = uct;
                best_child = child;
            }
        }
        return best_child;
    }
};

// --- MCTS Process ---
// Evaluates state and returns value from perspective of state.currentPlayer
float evaluateState(const GameState& state, std::vector<float>& policy) {
    std::vector<float> tensor = encodeBoard(state.M, state.N, state.board, state.currentPlayer, state.noX, state.noY);
    NetworkOutput out = forwardPass(tensor, state.M, state.N);
    policy = out.policy;
    return out.value;
}

void backpropagate(MCTSNode* node, float value) {
    // Value is from the perspective of the leaf node's currentPlayer.
    while (node != nullptr) {
        node->N_visits++;
        node->W_total_value += value;
        node->Q_mean_value = node->W_total_value / node->N_visits;

        // As we move up the tree, the perspective flips
        value = -value;
        node = node->parent;
    }
}

/*
	策略函数接口,该函数被对抗平台调用,每次传入当前状态,要求输出你的落子点,该落子点必须是一个符合游戏规则的落子点,不然对抗平台会直接认为你的程序有误
	
	input:
		为了防止对对抗平台维护的数据造成更改，所有传入的参数均为const属性
		M, N : 棋盘大小 M - 行数 N - 列数 均从0开始计， 左上角为坐标原点，行用x标记，列用y标记
		top : 当前棋盘每一列列顶的实际位置. e.g. 第i列为空,则_top[i] == M, 第i列已满,则_top[i] == 0
		_board : 棋盘的一维数组表示, 为了方便使用，在该函数刚开始处，我们已经将其转化为了二维数组board
				你只需直接使用board即可，左上角为坐标原点，数组从[0][0]开始计(不是[1][1])
				board[x][y]表示第x行、第y列的点(从0开始计)
				board[x][y] == 0/1/2 分别对应(x,y)处 无落子/有用户的子/有程序的子,不可落子点处的值也为0
		lastX, lastY : 对方上一次落子的位置, 你可能不需要该参数，也可能需要的不仅仅是对方一步的
				落子位置，这时你可以在自己的程序中记录对方连续多步的落子位置，这完全取决于你自己的策略
		noX, noY : 棋盘上的不可落子点(注:涫嫡饫锔?龅膖op已经替你处理了不可落子点，也就是说如果某一步
				所落的子的上面恰是不可落子点，那么UI工程中的代码就已经将该列的top值又进行了一次减一操作，
				所以在你的代码中也可以根本不使用noX和noY这两个参数，完全认为top数组就是当前每列的顶部即可,
				当然如果你想使用lastX,lastY参数，有可能就要同时考虑noX和noY了)
		以上参数实际上包含了当前状态(M N _top _board)以及历史信息(lastX lastY),你要做的就是在这些信息下给出尽可能明智的落子点
	output:
		你的落子点Point
*/
extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board,
						   const int lastX, const int lastY, const int noX, const int noY)
{
	/*
		不要更改这段代码
	*/
	int x = -1, y = -1; //最终将你的落子点存到x,y中
	int **board = new int *[M];
	for (int i = 0; i < M; i++)
	{
		board[i] = new int[N];
		for (int j = 0; j < N; j++)
		{
			board[i][j] = _board[i * N + j];
		}
	}

	/*
		根据你自己的策略来返回落子点,也就是根据你的策略完成对x,y的赋值
		该部分对参数使用没有限制，为了方便实现，你可以定义自己新的类、.h文件、.cpp文件
	*/
	//Add your own code below
	auto start_time = std::chrono::steady_clock::now();

	// Convert 2D board back to 1D for our GameState, though we could just pass 1D
	GameState rootState(M, N, _board, top, 2, noX, noY); // 2 means Program is to play

	MCTSNode* root = new MCTSNode(-1, rootState.currentPlayer, 1.0f, nullptr, rootState);

	// Initial evaluation
	std::vector<float> root_policy;
	evaluateState(rootState, root_policy);
	// We don't backpropagate here since root represents the current actual state,
	// but we could set root priors if we wanted. We'll set priors when children are created.

	int max_iterations = 10000;
	int iter = 0;

	for (; iter < max_iterations; ++iter) {
	    // Time Check (2.8 seconds), checked periodically to reduce overhead
	    if (iter % 128 == 0) {
	        auto current_time = std::chrono::steady_clock::now();
	        std::chrono::duration<double> elapsed = current_time - start_time;
	        if (elapsed.count() > 2.8) {
	            DEBUG_LOG("MCTS Interrupted at iteration: " << iter << " due to time limit (2.8s)");
	            break;
	        }
	    }

	    MCTSNode* node = root;
	    GameState state = rootState;

	    // Selection
	    while (!node->is_terminal && node->isFullyExpanded()) {
	        node = node->getBestUCTChild();
	        state.playMove(node->move);
	    }

	    // Expansion & Evaluation
	    float value = 0.0f;

	    if (node->is_terminal) {
	        // Terminal node evaluation (value is from perspective of node->currentPlayer)
	        if (node->terminal_value == node->currentPlayer) value = 1.0f; // Won
	        else if (node->terminal_value == 0) value = 0.0f; // Draw
	        else value = -1.0f; // Lost
	    } else {
	        // AlphaZero MCTS style Expansion:
	        // Evaluate the current node, then instantiate all its valid children with policy priors.
	        std::vector<float> policy;
	        value = evaluateState(state, policy); // Value from perspective of state.currentPlayer

	        for (int move : node->unexpanded_moves) {
	            GameState childState = state;
	            childState.playMove(move);
	            MCTSNode* child = new MCTSNode(move, childState.currentPlayer, policy[move], node, childState);
	            node->children.push_back(child);
	        }
	        node->unexpanded_moves.clear();
	    }

	    // Backpropagation
	    backpropagate(node, value);
	}

	DEBUG_LOG("MCTS Iterations completed: " << iter);

	// Select best move from root based on max visits
	int best_move = -1;
	int max_visits = -1;

	for (MCTSNode* child : root->children) {
	    if (child->N_visits > max_visits) {
	        max_visits = child->N_visits;
	        best_move = child->move;
	    }
	}

	if (best_move != -1) {
	    x = rootState.top[best_move] - 1;
	    y = best_move;
	    DEBUG_LOG("Selected move col: " << y << ", row: " << x << " with visits: " << max_visits);
	} else {
	    // Fallback if no children were explored
	    for (int i = N-1; i >= 0; i--) {
		    if (top[i] > 0) {
			    x = top[i] - 1;
			    y = i;
			    break;
		    }
	    }
	}

	delete root;

	/*
		不要更改这段代码
	*/
	clearArray(M, N, board);
	return new Point(x, y);
}

/*
	getPoint函数返回的Point指针是在本so模块中声明的，为避免产生堆错误，应在外部调用本so中的
	函数来释放空间，而不应该在外部直接delete
*/
extern "C" void clearPoint(Point *p)
{
	delete p;
	return;
}

/*
	清除top和board数组
*/
void clearArray(int M, int N, int **board)
{
	for (int i = 0; i < M; i++)
	{
		delete[] board[i];
	}
	delete[] board;
}
