#pragma once

#include <iostream>
#include <cstdint>

// #define DEBUG_MODE

#ifdef DEBUG_MODE
#define DEBUG_LOG(x)                 \
    do                               \
    {                                \
        std::cerr << x << std::endl; \
    } while (0)
#else
#define DEBUG_LOG(x) \
    do               \
    {                \
    } while (0)
#endif

const int PLAYER_COUNT = 2;
const int MAX_M = 12;
const int MAX_N = 12;
const int BITBOARD_WORDS = 3;
const int BITBOARD_COL_STRIDE = 16;
const int BITBOARD_VERTICAL_SHIFT = 1;
const int BITBOARD_HORIZONTAL_SHIFT = BITBOARD_COL_STRIDE;
const int BITBOARD_DIAG_UP_SHIFT = BITBOARD_COL_STRIDE + 1;
const int BITBOARD_DIAG_DOWN_SHIFT = BITBOARD_COL_STRIDE - 1;
const int CONNECT_TARGET = 4;
const int CONNECT_DIRECTION_COUNT = 4;
const int MAX_NODE_POOL = 300000;
const float C_PUCT = 1.0f;

// Zobrist 哈希表：2 个玩家，最多 12x12 棋盘。
extern uint64_t ZOBRIST[PLAYER_COUNT][MAX_M][MAX_N];
void initZobrist();

struct Bitboard
{
    uint64_t b[BITBOARD_WORDS];

    Bitboard() { b[0] = b[1] = b[2] = 0; }

    Bitboard operator|(const Bitboard &other) const
    {
        Bitboard res;
        res.b[0] = b[0] | other.b[0];
        res.b[1] = b[1] | other.b[1];
        res.b[2] = b[2] | other.b[2];
        return res;
    }

    Bitboard operator&(const Bitboard &other) const
    {
        Bitboard res;
        res.b[0] = b[0] & other.b[0];
        res.b[1] = b[1] & other.b[1];
        res.b[2] = b[2] & other.b[2];
        return res;
    }

    // 192-bit 位移，按 3 个 uint64_t 串接处理。
    Bitboard shift(int s) const
    {
        Bitboard res;
        if (s == 0)
            return *this;
        if (s > 0)
        {
            if (s < 64)
            {
                res.b[0] = b[0] << s;
                res.b[1] = (b[1] << s) | (b[0] >> (64 - s));
                res.b[2] = (b[2] << s) | (b[1] >> (64 - s));
            }
            else if (s == 64)
            {
                res.b[0] = 0;
                res.b[1] = b[0];
                res.b[2] = b[1];
            }
            else
            {
                // s > 64 的情况极少用到，可按需补充
            }
        }
        else
        {
            s = -s;
            if (s < 64)
            {
                res.b[0] = (b[0] >> s) | (b[1] << (64 - s));
                res.b[1] = (b[1] >> s) | (b[2] << (64 - s));
                res.b[2] = (b[2] >> s);
            }
            else if (s == 64)
            {
                res.b[0] = b[1];
                res.b[1] = b[2];
                res.b[2] = 0;
            }
            else
            {
                res.b[0] = 0;
                res.b[1] = 0;
                res.b[2] = 0;
            }
        }
        return res;
    }

    bool isEmpty() const
    {
        return b[0] == 0 && b[1] == 0 && b[2] == 0;
    }

    bool hasAlignedFour(int shift_step) const
    {
        Bitboard pair = (*this) & this->shift(shift_step);
        return !(pair & pair.shift(shift_step * 2)).isEmpty();
    }

    bool hasConnect4() const
    {
        return hasAlignedFour(BITBOARD_VERTICAL_SHIFT) ||
               hasAlignedFour(BITBOARD_HORIZONTAL_SHIFT) ||
               hasAlignedFour(BITBOARD_DIAG_UP_SHIFT) ||
               hasAlignedFour(BITBOARD_DIAG_DOWN_SHIFT);
    }

    static int cellIndex(int r, int c)
    {
        return c * BITBOARD_COL_STRIDE + r;
    }

    void setBit(int r, int c)
    {
        int idx = cellIndex(r, c);
        b[idx / 64] |= (1ULL << (idx % 64));
    }
};

struct GameState
{
    int M, N;
    int currentPlayer;
    int noX, noY;
    uint64_t hash_val;

    int top[MAX_N];
    Bitboard color[PLAYER_COUNT];

    GameState() {}

    GameState(int m, int n, const int *board_ptr, const int *top_ptr, int player, int nx, int ny)
    {
        M = m;
        N = n;
        currentPlayer = player;
        noX = nx;
        noY = ny;

        color[0] = Bitboard();
        color[1] = Bitboard();

        hash_val = 0;

        for (int c = 0; c < N; ++c)
        {
            top[c] = top_ptr[c];
            for (int r = top[c]; r < M; ++r)
            {
                if (r == noX && c == noY)
                    continue;

                int piece = board_ptr[r * N + c];
                if (piece == 1)
                {
                    color[0].setBit(r, c);
                    hash_val ^= ZOBRIST[0][r][c];
                }
                else if (piece == 2)
                {
                    color[1].setBit(r, c);
                    hash_val ^= ZOBRIST[1][r][c];
                }
            }
        }
    }

    void playMove(int col)
    {
        int r = top[col] - 1;

        color[currentPlayer - 1].setBit(r, col);
        hash_val ^= ZOBRIST[currentPlayer - 1][r][col];
        top[col]--;

        if (top[col] > 0 && top[col] - 1 == noX && col == noY)
        {
            top[col]--;
        }

        currentPlayer = (currentPlayer == 1) ? 2 : 1;
    }

    int checkTerminal() const
    {
        if (color[0].hasConnect4())
            return 1;
        if (color[1].hasConnect4())
            return 2;

        bool isFull = true;
        for (int c = 0; c < N; ++c)
        {
            if (top[c] > 0)
            {
                isFull = false;
                break;
            }
        }
        if (isFull)
            return 0;

        return -1;
    }
    int getPiece(int r, int c) const
    {
        int idx = Bitboard::cellIndex(r, c);
        if (color[0].b[idx / 64] & (1ULL << (idx % 64)))
            return 1;
        if (color[1].b[idx / 64] & (1ULL << (idx % 64)))
            return 2;
        return 0;
    }
};

struct MCTSNode
{
    int move, currentPlayer, N_visits, terminal_value;
    float W_total_value, Q_mean_value, P_prior, parity_bonus;
    int parent;

    int children[MAX_N];
    int num_children;
    int unexpanded_moves[MAX_N];
    int num_unexpanded;

    bool is_terminal;
    bool is_proven_win;
    bool is_proven_loss;
    bool all_legal_children_expanded;

    MCTSNode() {}

    void init(int m, int player, float prior, int p, GameState &state)
    {
        move = m;
        currentPlayer = player;
        P_prior = prior;
        parity_bonus = 0.0f;
        parent = p;
        N_visits = 0;
        W_total_value = 0.0f;
        Q_mean_value = 0.0f;
        is_terminal = false;
        is_proven_win = false;
        is_proven_loss = false;
        all_legal_children_expanded = false;
        terminal_value = -1;
        num_children = 0;
        num_unexpanded = 0;

        int term = state.checkTerminal();
        if (term != -1)
        {
            is_terminal = true;
            terminal_value = term;
            if (term != 0)
            {
                if (term == currentPlayer)
                    is_proven_win = true;
                else
                    is_proven_loss = true;
            }
        }
        else
        {
            int center = state.N / 2;
            for (int i = 0; i < state.N; ++i)
            {
                int offset = (i + 1) >> 1;
                int c = center + ((i & 1) ? -offset : offset);
                if (c >= 0 && c < state.N && state.top[c] > 0)
                    unexpanded_moves[num_unexpanded++] = c;
            }
        }
    }

    bool isFullyExpanded() const
    {
        return num_unexpanded == 0 || is_terminal || is_proven_win || is_proven_loss;
    }
};

extern thread_local MCTSNode g_node_pool[MAX_NODE_POOL];
extern thread_local int g_node_pool_size;

int allocateNode(int m, int player, float prior, int p, GameState &state);
int getBestUCTChild(int node_idx);
