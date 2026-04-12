#!/usr/bin/env python3
"""
MCTS Hex Agent

Monte Carlo Tree Search agent for the Hex game.
No swap rule handling (not required).

Usage:
    python3 gui_main.py --red-subprocess "python3 examples/python/mcts_agent.py"
    python3 gui_main.py --blue-subprocess "python3 examples/python/mcts_agent.py"
"""

import sys
import math
import random
import time

# Cell values
EMPTY = 0
RED = 1
BLUE = 2

# Hex grid neighbors: (delta_row, delta_col)
NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    __slots__ = ['parent', 'rank']

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        # Path splitting
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def connected(self, x, y):
        return self.find(x) == self.find(y)


# ---------------------------------------------------------------------------
# Hex Board
# ---------------------------------------------------------------------------

class HexBoard:
    """
    Board state with incremental win detection via Union-Find.

    Virtual nodes for edge connectivity:
      RED:  index n   = TOP edge,  index n+1 = BOTTOM edge
      BLUE: index n+2 = LEFT edge, index n+3 = RIGHT edge
    """
    __slots__ = [
        'size', 'board', 'uf_red', 'uf_blue', 'empty_cells',
        'RED_TOP', 'RED_BOTTOM', 'BLUE_LEFT', 'BLUE_RIGHT',
    ]

    def __init__(self, size):
        n = size * size
        self.size = size
        self.board = [EMPTY] * n
        self.uf_red  = UnionFind(n + 2)   # n=TOP,  n+1=BOTTOM
        self.uf_blue = UnionFind(n + 4)   # n+2=LEFT, n+3=RIGHT
        self.RED_TOP    = n
        self.RED_BOTTOM = n + 1
        self.BLUE_LEFT  = n + 2
        self.BLUE_RIGHT = n + 3
        self.empty_cells = list(range(n))

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def place(self, idx, color):
        """Place a stone at flat index idx."""
        size = self.size
        row, col = divmod(idx, size)
        self.board[idx] = color

        # When we place a stone on the board, that cell is no longer empty, remove it from the empty_cells
        # list.remove() is O(n). To speed up, use Swap-remove from empty_cells for O(1) removal
        ec = self.empty_cells
        pos = ec.index(idx)           # O(N) — acceptable for board sizes ≤ 21
        ec[pos] = ec[-1]
        ec.pop()

        if color == RED:
            uf = self.uf_red
            if row == 0:
                uf.union(idx, self.RED_TOP)
            if row == size - 1:
                uf.union(idx, self.RED_BOTTOM)
            for dr, dc in NEIGHBORS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nidx = nr * size + nc
                    if self.board[nidx] == RED:
                        uf.union(idx, nidx)
        else:  # BLUE
            uf = self.uf_blue
            if col == 0:
                uf.union(idx, self.BLUE_LEFT)
            if col == size - 1:
                uf.union(idx, self.BLUE_RIGHT)
            for dr, dc in NEIGHBORS:
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nidx = nr * size + nc
                    if self.board[nidx] == BLUE:
                        uf.union(idx, nidx)

    # ------------------------------------------------------------------
    # Win queries
    # ------------------------------------------------------------------

    def red_wins(self):
        return self.uf_red.connected(self.RED_TOP, self.RED_BOTTOM)

    def blue_wins(self):
        return self.uf_blue.connected(self.BLUE_LEFT, self.BLUE_RIGHT)

    def winner(self):
        if self.uf_red.connected(self.RED_TOP, self.RED_BOTTOM):
            return RED
        if self.uf_blue.connected(self.BLUE_LEFT, self.BLUE_RIGHT):
            return BLUE
        return None

    # ------------------------------------------------------------------
    # Deep copy
    # ------------------------------------------------------------------

    def copy(self):
        n = self.size * self.size
        new = HexBoard.__new__(HexBoard)
        new.size = self.size
        new.board = self.board[:]

        new.uf_red = UnionFind.__new__(UnionFind)
        new.uf_red.parent = self.uf_red.parent[:]
        new.uf_red.rank   = self.uf_red.rank[:]

        new.uf_blue = UnionFind.__new__(UnionFind)
        new.uf_blue.parent = self.uf_blue.parent[:]
        new.uf_blue.rank   = self.uf_blue.rank[:]

        new.empty_cells  = self.empty_cells[:]
        new.RED_TOP      = self.RED_TOP
        new.RED_BOTTOM   = self.RED_BOTTOM
        new.BLUE_LEFT    = self.BLUE_LEFT
        new.BLUE_RIGHT   = self.BLUE_RIGHT
        return new


# ---------------------------------------------------------------------------
# Fast rollout (no board copy needed)
# ---------------------------------------------------------------------------

def rollout(board, current_player):
    """
    Randomly assign remaining empty cells alternating between players,
    then determine the winner via BFS on RED's connectivity.

    No board mutation — works on a snapshot of board.board and board.empty_cells.
    Returns RED or BLUE.
    """
    size = board.size
    empty = board.empty_cells[:]      # copy of remaining moves
    random.shuffle(empty)

    assignment = board.board[:]       # copy of board cells
    player = current_player
    for idx in empty:
        assignment[idx] = player
        player = BLUE if player == RED else RED

    # BFS: check if RED has a path from row 0 to row size-1
    visited = bytearray(size * size)
    queue = []
    for col in range(size):
        idx = col   # row 0
        if assignment[idx] == RED and not visited[idx]:
            visited[idx] = 1
            queue.append(idx)

    head = 0
    while head < len(queue):
        idx = queue[head]
        head += 1
        row, col = divmod(idx, size)
        if row == size - 1:
            return RED
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                nidx = nr * size + nc
                if not visited[nidx] and assignment[nidx] == RED:
                    visited[nidx] = 1
                    queue.append(nidx)

    return BLUE   # Hex has no draws


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    """
    MCTS tree node.

    `wins` counts wins for the player who MOVED TO reach this node
    (= the opponent of current_player).  This keeps best_child() as a
    simple argmax everywhere in the tree.
    """
    __slots__ = [
        'board', 'parent', 'move', 'current_player',
        'children', 'visits', 'wins', 'untried_moves',
    ]

    def __init__(self, board, parent=None, move=None, current_player=RED):
        self.board          = board
        self.parent         = parent
        self.move           = move          # flat index of the move that led here
        self.current_player = current_player
        self.children       = []
        self.visits         = 0
        self.wins           = 0.0
        # Shuffle once at creation so expansion order is random
        self.untried_moves  = board.empty_cells[:]
        random.shuffle(self.untried_moves)

    def is_terminal(self):
        return self.board.winner() is not None

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def uct_value(self, log_parent, c):
        if self.visits == 0:
            return float('inf')
        return self.wins / self.visits + c * math.sqrt(log_parent / self.visits)

    def best_child(self, c=1.0):
        log_n = math.log(self.visits)
        return max(self.children, key=lambda ch: ch.uct_value(log_n, c))

    def expand(self):
        """Pop one untried move, create a child node, return it."""
        move       = self.untried_moves.pop()
        new_board  = self.board.copy()
        new_board.place(move, self.current_player)
        next_player = BLUE if self.current_player == RED else RED
        child = MCTSNode(
            board          = new_board,
            parent         = self,
            move           = move,
            current_player = next_player,
        )
        self.children.append(child)
        return child


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------

def mcts_search(board, my_color, time_limit_s):
    """
    Run MCTS from the given board state.
    Returns the best (row, col) move.
    """
    if not board.empty_cells:
        return (0, 0)

    root      = MCTSNode(board, current_player=my_color)
    deadline  = time.time() + time_limit_s
    iters     = 0

    while time.time() < deadline:
        # ---- Selection ----
        node = root
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child()

        # ---- Expansion ----
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # ---- Simulation ----
        if node.is_terminal():
            winner = node.board.winner()
        else:
            winner = rollout(node.board, node.current_player)

        # ---- Backpropagation ----
        cur = node
        while cur is not None:
            cur.visits += 1
            # mover = the player who moved TO reach cur = opponent of current_player
            mover = BLUE if cur.current_player == RED else RED
            if winner == mover:
                cur.wins += 1
            cur = cur.parent

        iters += 1

    print(f"DEBUG MCTS: {iters} iterations", file=sys.stderr)
    sys.stderr.flush()

    if not root.children:
        # No time for even one iteration — fall back to random
        idx = random.choice(board.empty_cells)
        return divmod(idx, board.size)

    # Pick child with most visits (most robust estimate)
    best = max(root.children, key=lambda ch: ch.visits)
    return divmod(best.move, board.size)


# ---------------------------------------------------------------------------
# Input parsing
# 11 RED 5:5:B,6:6:R,7:7:B
# ---------------------------------------------------------------------------

def parse_input(line):
    """
    Parse one line of the protocol.
    Returns (size, my_color, board).
    """
    parts = line.strip().split(maxsplit=2)
    size     = int(parts[0])
    my_color = RED if parts[1] == 'RED' else BLUE

    board = HexBoard(size)
    if len(parts) == 3 and parts[2].strip():
        for token in parts[2].split(','):
            r, c, color_char = token.split(':')
            color_val = RED if color_char == 'R' else BLUE
            board.place(int(r) * size + int(c), color_val)

    return size, my_color, board

# ---------------------------------------------------------------------------
# choose a move
# ---------------------------------------------------------------------------
def choose_move(size, board, my_color, time_limit):
    # DYNAMIC CENTER CALCULATION
    center = size // 2
    
    # 1. As RED: Always take center regardless of size
    if len(board.empty_cells) == size * size and my_color == RED:
        return (center, center)

    # 2. As BLUE: Dynamic Swap Logic
    if my_color == BLUE and len(board.empty_cells) == size * size - 1:
        c = board.board[center * size + center]
        # If RED took the center of this specific board size, SWAP
        if c == RED:
            return 'swap'
        else:
            return (center, center)

    return mcts_search(board, my_color, time_limit)


# ---------------------------------------------------------------------------
# Time budget per board size
# ---------------------------------------------------------------------------

# Use ~80 % of the allotted time to leave room for I/O and overhead.
_TIME_BUDGET = {
    11: 0.12,   # limit 150 ms → use 120 ms
    15: 0.16,   # limit 200 ms → use 160 ms
    19: 0.20,   # limit 250 ms → use 200 ms
    21: 0.24,   # limit 300 ms → use 240 ms
}
_DEFAULT_TIME = 0.08   # fallback for other sizes (default limit is 1 s)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    while True:
        try:
            line = input()
        except EOFError:
            break

        size, my_color, board = parse_input(line)
        time_limit = _TIME_BUDGET.get(size, _DEFAULT_TIME)
        result = choose_move(size, board, my_color, time_limit)
        print('swap' if result == 'swap' else f"{result[0]} {result[1]}")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
