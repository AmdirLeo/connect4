#include "AICore.h"
#include "Strategy.h"
#include "Judge.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

using namespace std;

uint64_t ZOBRIST[PLAYER_COUNT][MAX_M][MAX_N];
thread_local MCTSNode g_node_pool[MAX_NODE_POOL];
thread_local int g_node_pool_size = 0;

static constexpr uint64_t ZOBRIST_SEED = 123456789ULL;
static constexpr int MAX_MCTS_ITERATIONS = 500000;
static constexpr int TIME_CHECK_INTERVAL = 32;
static constexpr int ROLLOUT_MAX_PLIES = MAX_M * MAX_N;
static constexpr int SEARCH_PATH_CAPACITY = MAX_M * MAX_N + 1;
static constexpr double SEARCH_TIME_LIMIT_SEC = 2.70;
static constexpr float MIN_CHILD_PRIOR = 0.001f;
static constexpr float CENTER_PRIOR_SCALE = 0.02f;
static constexpr float PARITY_THREAT_BONUS = 0.05f;
static constexpr float DOUBLE_THREAT_BONUS = 0.50f;
static constexpr float EQUES_THREAT_BONUS = 0.80f;
static constexpr float PARITY_KNOTT_BONUS = 0.35f;
static constexpr float KNOTT_BONUS = 0.20f;
static constexpr float KNOTT_CANDIDATE_VALUE = 0.35f;
static const int CONNECT4_DIRS[CONNECT_DIRECTION_COUNT][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};

void initZobrist()
{
	std::mt19937_64 rng(ZOBRIST_SEED);
	for (int p = 0; p < PLAYER_COUNT; ++p)
	{
		for (int r = 0; r < MAX_M; ++r)
		{
			for (int c = 0; c < MAX_N; ++c)
			{
				ZOBRIST[p][r][c] = rng();
			}
		}
	}
}

static inline int otherPlayer(int player)
{
	return 3 - player;
}

static inline uint16_t columnMask(int col)
{
	return (uint16_t)(1u << col);
}

static inline bool isValidBoardSquare(const GameState &state, int r, int c)
{
	return r >= 0 && r < state.M && c >= 0 && c < state.N &&
		   !(r == state.noX && c == state.noY);
}

static inline bool isPlayableSquare(const GameState &state, int r, int c)
{
	return isValidBoardSquare(state, r, c) && state.top[c] > 0 && state.top[c] - 1 == r;
}

static bool wouldCompleteFourAt(const GameState &state, int player, int r, int c)
{
	if (!isValidBoardSquare(state, r, c) || state.getPiece(r, c) != 0)
		return false;

	for (int d = 0; d < CONNECT_DIRECTION_COUNT; ++d)
	{
		int count = 1;
		int dr = CONNECT4_DIRS[d][0], dc = CONNECT4_DIRS[d][1];

		for (int step = 1; step < CONNECT_TARGET; ++step)
		{
			int nr = r + dr * step, nc = c + dc * step;
			if (!isValidBoardSquare(state, nr, nc) || state.getPiece(nr, nc) != player)
				break;
			++count;
		}
		for (int step = 1; step < CONNECT_TARGET; ++step)
		{
			int nr = r - dr * step, nc = c - dc * step;
			if (!isValidBoardSquare(state, nr, nc) || state.getPiece(nr, nc) != player)
				break;
			++count;
		}
		if (count >= CONNECT_TARGET)
			return true;
	}
	return false;
}

static bool isColumnPoisoned(const GameState &state, int col)
{
	if (col < 0 || col >= state.N || state.top[col] <= 0)
		return false;

	int player = state.currentPlayer;
	int opponent = otherPlayer(player);

	GameState child = state;
	child.playMove(col);
	if (child.checkTerminal() == player || child.top[col] <= 0)
		return false;

	GameState reply = child;
	reply.playMove(col);
	return reply.checkTerminal() == opponent;
}

static uint16_t buildPoisonedMask(const GameState &state)
{
	uint16_t mask = 0;
	for (int c = 0; c < state.N; ++c)
	{
		if (isColumnPoisoned(state, c))
			mask |= columnMask(c);
	}
	return mask;
}

static bool hasTwoImmediateWins(const GameState &state, int player)
{
	int threats = 0;
	for (int c = 0; c < state.N; ++c)
	{
		if (state.top[c] <= 0)
			continue;

		GameState next = state;
		next.currentPlayer = player;
		next.playMove(c);
		if (next.checkTerminal() == player)
		{
			++threats;
			if (threats >= 2)
				return true;
		}
	}
	return false;
}

// 使用从底部开始的 1-based 行号：P1 控制奇数行，P2 控制偶数行。
// 换算到本工程的 top-origin 0-based r 后，M 为奇数时顶端坐标奇偶会翻转。
static inline bool playerControlsRow(const GameState &state, int player, int r)
{
	bool bottom_based_odd = ((state.M - r) & 1) != 0;
	return player == 1 ? bottom_based_odd : !bottom_based_odd;
}

static float parityThreatBonus(const GameState &state, int player)
{
	for (int r = 0; r < state.M; ++r)
	{
		for (int c = 0; c < state.N; ++c)
		{
			if (!isValidBoardSquare(state, r, c) || state.getPiece(r, c) != 0)
				continue;

			if (playerControlsRow(state, player, r) &&
				wouldCompleteFourAt(state, player, r, c))
			{
				return PARITY_THREAT_BONUS;
			}
		}
	}
	return 0.0f;
}

struct ThreatInfo
{
	Bitboard cells;
	uint16_t playable_cols;
	uint16_t knott_cols;
	uint16_t parity_knott_cols;
	int playable_count;
	int first_playable_col;
};

static inline bool bitboardHasCell(const Bitboard &bb, int r, int c)
{
	int idx = Bitboard::cellIndex(r, c);
	return (bb.b[idx / 64] & (1ULL << (idx % 64))) != 0;
}

static inline bool hasPlayableFork(const ThreatInfo &threats)
{
	return (threats.playable_cols & (uint16_t)(threats.playable_cols - 1)) != 0;
}

static inline bool hasVerticalThreatPair(const ThreatInfo &threats)
{
	return threats.knott_cols != 0;
}

static inline bool hasControlledVerticalThreatPair(const ThreatInfo &threats)
{
	return threats.parity_knott_cols != 0;
}

static ThreatInfo collectThreats(const GameState &state, int player)
{
	ThreatInfo info;
	info.cells = Bitboard();
	info.playable_cols = 0;
	info.knott_cols = 0;
	info.parity_knott_cols = 0;
	info.playable_count = 0;
	info.first_playable_col = -1;

	for (int c = 0; c < state.N; ++c)
	{
		for (int r = 0; r < state.M; ++r)
		{
			if (!isValidBoardSquare(state, r, c) || state.getPiece(r, c) != 0)
				continue;

			if (!wouldCompleteFourAt(state, player, r, c))
				continue;

			info.cells.setBit(r, c);
			if (isPlayableSquare(state, r, c))
			{
				info.playable_cols |= columnMask(c);
				++info.playable_count;
				if (info.first_playable_col < 0)
					info.first_playable_col = c;
			}
		}
	}

	for (int c = 0; c < state.N; ++c)
	{
		for (int r = 1; r < state.M; ++r)
		{
			if (!bitboardHasCell(info.cells, r, c) || !bitboardHasCell(info.cells, r - 1, c))
				continue;

			info.knott_cols |= columnMask(c);
			if (playerControlsRow(state, player, r - 1))
				info.parity_knott_cols |= columnMask(c);
		}
	}

	return info;
}

static void resetSearchStorage()
{
	g_node_pool_size = 0;
}

static int chooseExpandLimit(const GameState &state, int legal_moves)
{
	if (legal_moves <= 4)
		return legal_moves;

	int remaining_slots = 0;
	for (int c = 0; c < state.N; ++c)
		remaining_slots += state.top[c];

	if (remaining_slots <= state.N * 3)
		return std::min(legal_moves, 6);
	return std::min(legal_moves, 4);
}

static int countImmediateWinsForPlayer(const GameState &state, int player)
{
	int wins = 0;
	for (int c = 0; c < state.N; ++c)
	{
		if (state.top[c] <= 0)
			continue;

		GameState next = state;
		next.currentPlayer = player;
		next.playMove(c);
		if (next.checkTerminal() == player)
			wins++;
	}
	return wins;
}

static int findImmediateWinningMove(const GameState &state, int player)
{
	for (int c = 0; c < state.N; ++c)
	{
		if (state.top[c] <= 0)
			continue;

		GameState next = state;
		next.currentPlayer = player;
		next.playMove(c);
		if (next.checkTerminal() == player)
			return c;
	}
	return -1;
}

static int findForcedBlockMove(const GameState &state, int *block_count)
{
	int opponent = otherPlayer(state.currentPlayer);
	int block_move = -1;
	*block_count = 0;

	for (int c = 0; c < state.N; ++c)
	{
		if (state.top[c] <= 0)
			continue;

		GameState oppState = state;
		oppState.currentPlayer = opponent;
		oppState.playMove(c);
		if (oppState.checkTerminal() == opponent)
		{
			block_move = c;
			++(*block_count);
		}
	}
	return block_move;
}

static bool moveAllowsImmediateLoss(const GameState &state, int move)
{
	if (move < 0 || move >= state.N || state.top[move] <= 0)
		return true;

	GameState childState = state;
	int player = state.currentPlayer;
	childState.playMove(move);
	if (childState.checkTerminal() == player)
		return false;

	return countImmediateWinsForPlayer(childState, childState.currentPlayer) > 0;
}

static bool isForcedWinInTwo(const GameState &state, int move)
{
	if (move < 0 || move >= state.N || state.top[move] <= 0)
		return false;

	int player = state.currentPlayer;
	GameState childState = state;
	childState.playMove(move);
	if (childState.checkTerminal() == player)
		return true;

	int opponent = childState.currentPlayer;
	if (countImmediateWinsForPlayer(childState, opponent) > 0)
		return false;

	bool opponent_has_reply = false;
	for (int c = 0; c < childState.N; ++c)
	{
		if (childState.top[c] <= 0)
			continue;

		opponent_has_reply = true;
		GameState replyState = childState;
		replyState.playMove(c);
		if (replyState.checkTerminal() == opponent)
			return false;
		if (countImmediateWinsForPlayer(replyState, player) == 0)
			return false;
	}

	return opponent_has_reply;
}

static int findThreatSpaceMove(const GameState &state, int player, bool *is_proven)
{
	int best_knott_move = -1;
	int best_parity_knott_move = -1;
	*is_proven = false;

	for (int c = 0; c < state.N; ++c)
	{
		if (state.top[c] <= 0)
			continue;

		GameState child = state;
		child.currentPlayer = player;
		child.playMove(c);

		if (child.checkTerminal() == player)
		{
			*is_proven = true;
			return c;
		}

		if (countImmediateWinsForPlayer(child, child.currentPlayer) > 0)
			continue;

		ThreatInfo threats = collectThreats(child, player);
		if (hasPlayableFork(threats))
		{
			*is_proven = true;
			return c;
		}

		if (best_parity_knott_move < 0 && hasControlledVerticalThreatPair(threats))
			best_parity_knott_move = c;
		else if (best_knott_move < 0 && hasVerticalThreatPair(threats))
			best_knott_move = c;
	}

	if (best_parity_knott_move >= 0)
		return best_parity_knott_move;
	return best_knott_move;
}

static uint32_t nextRolloutRandom(uint64_t salt)
{
	static thread_local uint32_t rng_state = 0x9e3779b9u;
	rng_state ^= (uint32_t)salt + 0x85ebca6bu + (rng_state << 6) + (rng_state >> 2);
	rng_state ^= rng_state << 13;
	rng_state ^= rng_state >> 17;
	rng_state ^= rng_state << 5;
	return rng_state;
}

static float simulatePlayout(GameState state);

static inline float simulateFromState(const GameState &state)
{
	return simulatePlayout(state);
}

// 智能 rollout：立即赢 > 立即挡 > 随机安全列；全是 poisoned 时才允许 poisoned。
static float simulatePlayout(GameState state)
{
	int root_player = state.currentPlayer;

	for (int ply = 0; ply < ROLLOUT_MAX_PLIES; ++ply)
	{
		int term = state.checkTerminal();
		if (term != -1)
		{
			if (term == 0)
				return 0.0f;
			return term == root_player ? 1.0f : -1.0f;
		}

		int player = state.currentPlayer;
		int move = findImmediateWinningMove(state, player);
		if (move == -1)
		{
			int block_count = 0;
			move = findForcedBlockMove(state, &block_count);
		}

		if (move == -1)
		{
			int legal_moves[MAX_N];
			int safe_moves[MAX_N];
			int legal_count = 0;
			int safe_count = 0;
			uint16_t poisoned = buildPoisonedMask(state);

			for (int c = 0; c < state.N; ++c)
			{
				if (state.top[c] <= 0)
					continue;

				legal_moves[legal_count++] = c;
				if ((poisoned & columnMask(c)) == 0)
					safe_moves[safe_count++] = c;
			}

			if (legal_count == 0)
				return 0.0f;

			if (safe_count > 0)
				move = safe_moves[nextRolloutRandom(state.hash_val) % (uint32_t)safe_count];
			else
				move = legal_moves[nextRolloutRandom(state.hash_val) % (uint32_t)legal_count];
		}

		state.playMove(move);
	}

	return 0.0f;
}

static float rankExpansionMove(const GameState &state, int move, uint16_t parity_mask)
{
	GameState child = state;
	child.playMove(move);

	float score = 1.0f;
	int center_delta = std::abs((move << 1) - (state.N - 1));
	score += CENTER_PRIOR_SCALE * (float)(state.N - center_delta);
	if (parity_mask & columnMask(move))
		score += parityThreatBonus(child, state.currentPlayer);

	if (hasTwoImmediateWins(child, state.currentPlayer))
		score += DOUBLE_THREAT_BONUS;

	ThreatInfo threats = collectThreats(child, state.currentPlayer);
	if (hasPlayableFork(threats))
		score += EQUES_THREAT_BONUS;
	if (hasControlledVerticalThreatPair(threats))
		score += PARITY_KNOTT_BONUS;
	else if (hasVerticalThreatPair(threats))
		score += KNOTT_BONUS;

	return score > MIN_CHILD_PRIOR ? score : MIN_CHILD_PRIOR;
}

static void addExpandedChildMasked(int curr_idx, const GameState &state, int move, float prior, uint16_t parity_mask)
{
	MCTSNode &node = g_node_pool[curr_idx];
	if (node.num_children >= MAX_N)
		return;

	GameState childState = state;
	childState.playMove(move);
	int child_idx = allocateNode(move, childState.currentPlayer, prior, curr_idx, childState);
	if (child_idx >= 0)
	{
		if ((parity_mask & columnMask(move)) == 0)
			g_node_pool[child_idx].parity_bonus = 0.0f;
		node.children[node.num_children++] = child_idx;
	}
}

static void addExpandedChild(int curr_idx, const GameState &state, int move, float prior)
{
	addExpandedChildMasked(curr_idx, state, move, prior, columnMask(move));
}

static void expandForcedMove(int curr_idx, const GameState &state, int move)
{
	MCTSNode &node = g_node_pool[curr_idx];
	node.unexpanded_moves[0] = move;
	node.num_unexpanded = 1;
	addExpandedChild(curr_idx, state, move, 1.0f);
	node.num_unexpanded = 0;
	node.all_legal_children_expanded = true;
}

static inline float provenValue(const MCTSNode &node)
{
	if (node.is_proven_win)
		return 1.0f;
	if (node.is_proven_loss)
		return -1.0f;
	if (node.is_terminal)
	{
		if (node.terminal_value == 0)
			return 0.0f;
		return node.terminal_value == node.currentPlayer ? 1.0f : -1.0f;
	}
	return 0.0f;
}

static inline void refreshProvenFromChildren(int node_idx)
{
	MCTSNode &node = g_node_pool[node_idx];
	if (node.is_terminal || node.is_proven_win || node.is_proven_loss)
		return;

	bool has_child = false;
	bool all_children_win_for_child_player = true;
	for (int i = 0; i < node.num_children; ++i)
	{
		MCTSNode &child = g_node_pool[node.children[i]];
		has_child = true;

		if (child.is_proven_loss)
		{
			node.is_proven_win = true;
			return;
		}

		if (!child.is_proven_win)
			all_children_win_for_child_player = false;
	}

	if (has_child && node.all_legal_children_expanded && all_children_win_for_child_player)
		node.is_proven_loss = true;
}

static void expandRankedChildren(int curr_idx, const GameState &state, int path_depth)
{
	MCTSNode &node = g_node_pool[curr_idx];
	int moves[MAX_N];
	float scores[MAX_N];
	bool chosen[MAX_N] = {false};
	int count = 0;

	uint16_t legal_mask = 0;
	for (int i = 0; i < node.num_unexpanded; ++i)
	{
		int move = node.unexpanded_moves[i];
		if (state.top[move] > 0)
			legal_mask |= columnMask(move);
	}

	uint16_t poisoned_mask = buildPoisonedMask(state) & legal_mask;
	uint16_t valid_moves_mask = legal_mask & (uint16_t)(~poisoned_mask);
	bool all_poisoned = (valid_moves_mask == 0);
	uint16_t active_mask = all_poisoned ? legal_mask : valid_moves_mask;
	uint16_t parity_mask = all_poisoned ? 0 : valid_moves_mask;

	for (int i = 0; i < node.num_unexpanded; ++i)
	{
		int move = node.unexpanded_moves[i];
		if ((active_mask & columnMask(move)) == 0)
			continue;

		moves[count] = move;
		scores[count] = rankExpansionMove(state, move, parity_mask);
		++count;
	}

	if (count == 0)
	{
		node.num_unexpanded = 0;
		node.all_legal_children_expanded = true;
		return;
	}

	int limit = (path_depth == 1) ? count : chooseExpandLimit(state, count);
	if (limit > count)
		limit = count;

	float selected_sum = 0.0f;
	for (int pick = 0; pick < limit; ++pick)
	{
		int best = -1;
		float best_score = -1.0f;
		for (int i = 0; i < count; ++i)
		{
			if (!chosen[i] && scores[i] > best_score)
			{
				best_score = scores[i];
				best = i;
			}
		}
		if (best == -1)
			break;
		chosen[best] = true;
		selected_sum += scores[best];
	}

	if (selected_sum <= 0.0f)
		selected_sum = 1.0f;

	for (int i = 0; i < count; ++i)
	{
		if (chosen[i])
			addExpandedChildMasked(curr_idx, state, moves[i], scores[i] / selected_sum, parity_mask);
	}

	node.num_unexpanded = 0;
	node.all_legal_children_expanded = (limit == count);
	refreshProvenFromChildren(curr_idx);
}

struct ExpansionTactic
{
	int move;
	float value;
	bool handled;
	bool proven_win;
	bool proven_loss;
};

static ExpansionTactic noExpansionTactic()
{
	ExpansionTactic tactic;
	tactic.move = -1;
	tactic.value = 0.0f;
	tactic.handled = false;
	tactic.proven_win = false;
	tactic.proven_loss = false;
	return tactic;
}

static inline bool passesPoisonFilter(uint16_t poisoned_mask, int move, bool all_poisoned)
{
	return all_poisoned || (poisoned_mask & columnMask(move)) == 0;
}

static int countSafeUnexpandedMoves(const MCTSNode &node, uint16_t poisoned_mask)
{
	int safe_count = 0;
	for (int i = 0; i < node.num_unexpanded; ++i)
	{
		if ((poisoned_mask & columnMask(node.unexpanded_moves[i])) == 0)
			++safe_count;
	}
	return safe_count;
}

static int findImmediateWinInUnexpanded(const MCTSNode &node, const GameState &state, int player)
{
	for (int i = 0; i < node.num_unexpanded; ++i)
	{
		int move = node.unexpanded_moves[i];
		GameState child = state;
		child.playMove(move);
		if (child.checkTerminal() == player)
			return move;
	}
	return -1;
}

static int findDoubleThreatInUnexpanded(const MCTSNode &node, const GameState &state, int player,
										uint16_t poisoned_mask, bool all_poisoned)
{
	for (int i = 0; i < node.num_unexpanded; ++i)
	{
		int move = node.unexpanded_moves[i];
		if (!passesPoisonFilter(poisoned_mask, move, all_poisoned))
			continue;

		GameState child = state;
		child.playMove(move);
		if (hasTwoImmediateWins(child, player))
			return move;
	}
	return -1;
}

static ExpansionTactic chooseExpansionTactic(const MCTSNode &node, const GameState &state)
{
	ExpansionTactic tactic = noExpansionTactic();
	int player = state.currentPlayer;
	uint16_t poisoned = buildPoisonedMask(state);
	int safe_count = countSafeUnexpandedMoves(node, poisoned);
	bool all_poisoned = (safe_count == 0);

	int win_move = findImmediateWinInUnexpanded(node, state, player);
	if (win_move != -1)
	{
		tactic.move = win_move;
		tactic.value = 1.0f;
		tactic.handled = true;
		tactic.proven_win = true;
		return tactic;
	}

	int block_count = 0;
	int block_move = findForcedBlockMove(state, &block_count);
	if (block_move != -1)
	{
		if (block_count >= 2)
		{
			tactic.value = -1.0f;
			tactic.handled = true;
			tactic.proven_loss = true;
			return tactic;
		}
		tactic.move = block_move;
		tactic.value = simulateFromState(state);
		tactic.handled = true;
		return tactic;
	}

	int double_threat_move = findDoubleThreatInUnexpanded(node, state, player, poisoned, all_poisoned);
	bool threat_space_proven = false;
	int threat_space_move = findThreatSpaceMove(state, player, &threat_space_proven);
	if (threat_space_move != -1)
	{
		tactic.move = threat_space_move;
		tactic.handled = true;
		if (threat_space_proven)
		{
			tactic.value = 1.0f;
			tactic.proven_win = true;
			return tactic;
		}
		tactic.value = KNOTT_CANDIDATE_VALUE;
		return tactic;
	}

	if (double_threat_move != -1)
	{
		tactic.move = double_threat_move;
		tactic.value = 1.0f;
		tactic.handled = true;
		tactic.proven_win = true;
		return tactic;
	}

	return tactic;
}

static float expandAndEvaluateNode(int curr_idx, GameState &state, int path_depth)
{
	MCTSNode &node = g_node_pool[curr_idx];
	if (node.is_terminal || node.is_proven_win || node.is_proven_loss)
		return provenValue(node);

	ExpansionTactic tactic = chooseExpansionTactic(node, state);
	if (tactic.handled)
	{
		if (tactic.move >= 0)
			expandForcedMove(curr_idx, state, tactic.move);
		node.is_proven_win = tactic.proven_win;
		node.is_proven_loss = tactic.proven_loss;
		return tactic.value;
	}

	float value = simulateFromState(state);
	expandRankedChildren(curr_idx, state, path_depth);
	return value;
}

static int getRootChildVisits(int root_idx, int move)
{
	for (int i = 0; i < g_node_pool[root_idx].num_children; ++i)
	{
		int child_idx = g_node_pool[root_idx].children[i];
		if (g_node_pool[child_idx].move == move)
			return g_node_pool[child_idx].N_visits;
	}
	return 0;
}

static int chooseRootMoveWithTacticalCheck(const GameState &rootState, int root_idx)
{
	int best_move = -1;
	int max_visits = -1;

	for (int i = 0; i < g_node_pool[root_idx].num_children; ++i)
	{
		int child_idx = g_node_pool[root_idx].children[i];
		if (g_node_pool[child_idx].N_visits > max_visits)
		{
			max_visits = g_node_pool[child_idx].N_visits;
			best_move = g_node_pool[child_idx].move;
		}
	}

	if (best_move == -1)
		return -1;

	int player = rootState.currentPlayer;
	int best_immediate_win = -1;
	int best_immediate_visits = -1;
	int best_forced_win = -1;
	int best_forced_visits = -1;
	int best_safe_move = -1;
	int best_safe_visits = -1;

	for (int move = 0; move < rootState.N; ++move)
	{
		if (rootState.top[move] <= 0)
			continue;

		int visits = getRootChildVisits(root_idx, move);

		GameState childState = rootState;
		childState.playMove(move);
		if (childState.checkTerminal() == player)
		{
			if (visits > best_immediate_visits)
			{
				best_immediate_visits = visits;
				best_immediate_win = move;
			}
			continue;
		}

		if (!moveAllowsImmediateLoss(rootState, move))
		{
			if (visits > best_safe_visits)
			{
				best_safe_visits = visits;
				best_safe_move = move;
			}

			if (visits * 2 >= max_visits && isForcedWinInTwo(rootState, move) &&
				visits > best_forced_visits)
			{
				best_forced_visits = visits;
				best_forced_win = move;
			}
		}
	}

	if (best_immediate_win != -1)
		return best_immediate_win;
	if (best_forced_win != -1)
		return best_forced_win;
	if (moveAllowsImmediateLoss(rootState, best_move) && best_safe_move != -1)
		return best_safe_move;
	return best_move;
}

int allocateNode(int m, int player, float prior, int p, GameState &state)
{
	if (g_node_pool_size >= MAX_NODE_POOL)
		return -1;

	int idx = g_node_pool_size++;
	g_node_pool[idx].init(m, player, prior, p, state);
	if (m != -1)
		g_node_pool[idx].parity_bonus = parityThreatBonus(state, otherPlayer(state.currentPlayer));
	return idx;
}

int getBestUCTChild(int node_idx)
{
	int best_child = -1;
	float best_uct = -1e9f;
	MCTSNode &node = g_node_pool[node_idx];

	for (int i = 0; i < node.num_children; ++i)
	{
		int child_idx = node.children[i];
		if (g_node_pool[child_idx].is_proven_loss)
			return child_idx;
	}

	for (int i = 0; i < node.num_children; ++i)
	{
		int child_idx = node.children[i];
		MCTSNode &child = g_node_pool[child_idx];
		if (child.is_proven_win)
			continue;

		float uct;
		if (child.N_visits == 0)
		{
			uct = C_PUCT * child.P_prior * std::sqrt((float)node.N_visits + 1e-8f) + child.parity_bonus;
		}
		else
		{
			float q_val = -child.Q_mean_value;
			float u_val = C_PUCT * child.P_prior * std::sqrt((float)node.N_visits) / (1.0f + child.N_visits);
			uct = q_val + u_val + child.parity_bonus;
		}
		if (uct > best_uct)
		{
			best_uct = uct;
			best_child = child_idx;
		}
	}
	return best_child >= 0 ? best_child : (node.num_children > 0 ? node.children[0] : -1);
}

static int computeLandingRow(int col, const GameState &state)
{
	int row = state.top[col] - 1;
	if (col == state.noY && row == state.noX)
	{
		row--;
	}
	return row;
}

static bool chooseFallbackMove(int &x, int &y, int M, int N, const int *top, int noX, int noY)
{
	for (int i = N - 1; i >= 0; --i)
	{
		if (top[i] <= 0)
			continue;

		int row = top[i] - 1;
		if (i == noY && row == noX)
		{
			row--;
		}
		if (row >= 0 && row < M)
		{
			x = row;
			y = i;
			return true;
		}
	}
	return false;
}

extern "C" Point *getPoint(const int M, const int N, const int *top, const int *_board, const int lastX,
							const int lastY, const int noX, const int noY)
{
	auto start_time = std::chrono::steady_clock::now();
	(void)lastX;
	(void)lastY;

	static bool zobrist_initialized = false;
	if (!zobrist_initialized)
	{
		initZobrist();
		zobrist_initialized = true;
	}

	GameState rootState(M, N, _board, top, 2, noX, noY);
	resetSearchStorage();
	int root_idx = allocateNode(-1, rootState.currentPlayer, 1.0f, -1, rootState);

	int max_iterations = MAX_MCTS_ITERATIONS;
	int iter = 0;

	for (; iter < max_iterations; ++iter)
	{
		if (iter % TIME_CHECK_INTERVAL == 0)
		{
			auto current_time = std::chrono::steady_clock::now();
			std::chrono::duration<double> elapsed = current_time - start_time;
			if (elapsed.count() > SEARCH_TIME_LIMIT_SEC)
				break;
		}

		int curr_idx = root_idx;
		GameState state = rootState;

		int search_path[SEARCH_PATH_CAPACITY];
		int path_depth = 0;
		search_path[path_depth++] = curr_idx;

		// Selection
		while (!g_node_pool[curr_idx].is_terminal &&
			   !g_node_pool[curr_idx].is_proven_win &&
			   !g_node_pool[curr_idx].is_proven_loss &&
			   g_node_pool[curr_idx].isFullyExpanded())
		{
			int next_idx = getBestUCTChild(curr_idx);
			if (next_idx < 0)
				break;
			curr_idx = next_idx;
			state.playMove(g_node_pool[curr_idx].move);

			search_path[path_depth++] = curr_idx;
		}

		float value = expandAndEvaluateNode(curr_idx, state, path_depth);

		for (int i = path_depth - 1; i >= 0; --i)
		{
			int n_idx = search_path[i];
			MCTSNode &n = g_node_pool[n_idx];
			n.N_visits++;
			n.W_total_value += value;
			n.Q_mean_value = n.W_total_value / n.N_visits;
			refreshProvenFromChildren(n_idx);
			value = -value;
		}
	}

	int best_move = chooseRootMoveWithTacticalCheck(rootState, root_idx);

	int x = -1, y = -1;
	if (best_move != -1)
	{
		int row = computeLandingRow(best_move, rootState);
		if (row >= 0 && row < M)
		{
			x = row;
			y = best_move;
		}
		else
		{
			best_move = -1;
		}
	}

	if (best_move == -1)
	{
		if (!chooseFallbackMove(x, y, M, N, top, noX, noY))
		{
			for (int i = N - 1; i >= 0; i--)
			{
				if (top[i] > 0)
				{
					x = top[i] - 1;
					y = i;
					break;
				}
			}
		}
	}

	return new Point(x, y);
}

extern "C" void clearPoint(Point *p) { delete p; }
