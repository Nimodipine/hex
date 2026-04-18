#!/usr/bin/env python3
"""
MCTS Hex Agent

Monte Carlo Tree Search agent for the Hex game.
No swap rule handling (not required).

Usage:
    python3 gui_main.py --red-subprocess "python3 examples/python/mcts_random_agent.py"
    python3 gui_main.py --blue-subprocess "python3 examples/python/mcts_random_agent.py"
"""

import sys
import math
import random
import time
import gc
from collections import deque

# Cell values
EMPTY = 0
RED = 1
BLUE = 2

# Hex grid neighbors: (delta_row, delta_col)
NEIGHBORS = [(-1, 0), (-1, 1), (0, 1), (1, 0), (1, -1), (0, -1)]

# Bridge patterns: (dr_B, dc_B, dr_p1, dc_p1, dr_p2, dc_p2)
#   B = partner stone offset from anchor A, p1/p2 = the two shared corridor cells
BRIDGE_PATTERNS = [
    ( 1,  1,  0,  1,  1,  0),
    (-1,  2, -1,  1,  0,  1),
    (-2,  1, -1,  0, -1,  1),
    (-1, -1,  0, -1, -1,  0),
    ( 1, -2,  1, -1,  0, -1),
    ( 2, -1,  1,  0,  1, -1),
]


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
# Cache of pre-computed neighbor tables keyed by board size.
# Computed once per size and reused across all rollout calls.
_ADJ_CACHE = {}   # size -> (ADJ, TOP_CELLS, BOTTOM_START)

def _get_rollout_tables(size):
    """Return pre-computed BFS helpers for the given board size."""
    if size not in _ADJ_CACHE:
        # ADJ[idx] = list of valid neighbor flat-indices for cell idx
        ADJ = []
        for idx in range(size * size):
            r, c = divmod(idx, size)
            nbrs = []
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nbrs.append(nr * size + nc)
            ADJ.append(nbrs)
        TOP_CELLS    = list(range(size))          # seed cells for BFS (row 0)
        BOTTOM_START = (size - 1) * size          # idx >= BOTTOM_START  ↔  row == size-1
        _ADJ_CACHE[size] = (ADJ, TOP_CELLS, BOTTOM_START)
    return _ADJ_CACHE[size]



def rollout(board, current_player, deadline):
    """
    Randomly assign remaining empty cells alternating between players,
    then determine the winner via BFS on RED's connectivity.

    No board mutation — works on a snapshot of board.board and board.empty_cells.
    Returns RED, BLUE, or None if the deadline was exceeded mid-rollout.
    """
    size  = board.size
    ADJ, TOP_CELLS, BOTTOM_START = _get_rollout_tables(size)

    empty = random.sample(board.empty_cells, len(board.empty_cells))

    assignment = bytearray(board.board)   # fast byte array copy
    player = current_player
    for idx in empty:
        assignment[idx] = player
        player = BLUE if player == RED else RED

    # BFS with pre-computed neighbor table
    visited = bytearray(size * size)
    queue   = []
    for idx in TOP_CELLS:                 # pre-computed top row
        if assignment[idx] == RED:
            visited[idx] = 1
            queue.append(idx)

    head = 0
    while head < len(queue):
        if deadline is not None and head % 32 == 0 and time.time() >= deadline:
            return None
        idx = queue[head]; head += 1
        if idx >= BOTTOM_START:           # reached last row → RED wins
            return RED
        for nidx in ADJ[idx]:            # pre-computed neighbor list
            if not visited[nidx] and assignment[nidx] == RED:
                visited[nidx] = 1
                queue.append(nidx)

    return BLUE


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
        if time.time() >= deadline:
            # print(f"selection: {time.time() - deadline}", file=sys.stderr)
            # sys.stderr.flush()
            break
        # ---- Expansion ----
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()
        if time.time() >= deadline:
            # print(f"expansion: {time.time() - deadline}", file=sys.stderr)
            # sys.stderr.flush()
            break
        # ---- Simulation ----
        if node.is_terminal():
            winner = node.board.winner()
        else:
            winner = rollout(node.board, node.current_player, deadline)
        if time.time() >= deadline:
            # print(f"simulation: {time.time() - deadline}", file=sys.stderr)
            # sys.stderr.flush()
            break
        # ---- Backpropagation ----
        cur = node
        while cur is not None and winner is not None:
            if time.time() >= deadline:
                # print(f"backpropagation: {time.time() - deadline}", file=sys.stderr)
                # sys.stderr.flush()
                break
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
# Cut cells: empty cells on every shortest resistance path of the opponent
# ---------------------------------------------------------------------------

def _bfs_resistance(board_arr, size, opp, my_color, reverse=False):
    """
    0-1 BFS computing the minimum resistance distance from the opponent's
    start edge (reverse=False) or goal edge (reverse=True) to every cell.

    Cost model:
      opponent stone → 0   (already traversed)
      empty cell     → 1   (needs to be filled)
      our stone      → INF (impassable)

    Returns a list of length size*size with distances.
    """
    INF = float('inf')
    dist = [INF] * (size * size)
    dq = deque()

    # Determine seed edge: RED goes top→bottom, BLUE goes left→right
    # reverse=True means we seed from the goal edge instead
    for i in range(size):
        if opp == RED:
            idx = (size - 1) * size + i if reverse else i   # last row / row 0
        else:
            idx = i * size + (size - 1) if reverse else i * size  # last col / col 0

        if board_arr[idx] == my_color:
            continue
        cost = 0 if board_arr[idx] == opp else 1
        if cost < dist[idx]:
            dist[idx] = cost
            dq.appendleft(idx) if cost == 0 else dq.append(idx)

    while dq:
        u = dq.popleft()
        ru, cu = divmod(u, size)
        for dr, dc in NEIGHBORS:
            nr, nc = ru + dr, cu + dc
            if not (0 <= nr < size and 0 <= nc < size):
                continue
            v = nr * size + nc
            if board_arr[v] == my_color:
                continue
            edge = 0 if board_arr[v] == opp else 1
            nd = dist[u] + edge
            if nd < dist[v]:
                dist[v] = nd
                dq.appendleft(v) if edge == 0 else dq.append(v)

    return dist


def cut_cells(board, my_color):
    """
    Return a list of empty cell indices that lie on at least one of the
    opponent's shortest-resistance paths.

    Algorithm (forward + backward 0-1 BFS):
      1. dist_fwd[v]  = resistance from start edge to v
      2. dist_bwd[v]  = resistance from goal  edge to v
      3. shortest     = min resistance across the goal edge cells
      4. v is a cut cell if it is EMPTY and:
             dist_fwd[v] + 1 + dist_bwd[v] == shortest

    The '+1' is the cost of placing at v (empty → cost 1).

    Returns a list of flat indices (empty cells only).
    """
    size = board.size
    opp  = BLUE if my_color == RED else RED
    board_arr = board.board
    INF = float('inf')

    dist_fwd = _bfs_resistance(board_arr, size, opp, my_color, reverse=False)
    dist_bwd = _bfs_resistance(board_arr, size, opp, my_color, reverse=True)

    # Shortest path length = minimum dist_fwd at the goal edge
    shortest = INF
    for i in range(size):
        idx = (size - 1) * size + i if opp == RED else i * size + (size - 1)
        if dist_fwd[idx] < shortest:
            shortest = dist_fwd[idx]

    if shortest == INF:
        return []   # opponent already has no path (we've won, or board full)

    cuts = []
    for idx in range(size * size):
        if board_arr[idx] != EMPTY:
            continue
        if dist_fwd[idx] + dist_bwd[idx] - 1 == shortest:
            cuts.append(idx)

    return cuts


# ---------------------------------------------------------------------------
# Bridge move with middle-area fallback
# ---------------------------------------------------------------------------

def bridge_move_random(board, my_color):
    """
    Find a random empty cell that forms a bridge with an existing friendly stone.

    A bridge between anchor A and candidate B requires:
      - A is already my_color
      - B is empty
      - both corridor cells p1, p2 are also empty

    Fallback: if no friendly stones exist yet (or no bridge candidates found),
    pick a random empty cell inside the middle third of the board.
    """
    size = board.size
    board_arr = board.board
    candidates = set()

    for idx in range(size * size):
        if board_arr[idx] != my_color:
            continue
        r, c = divmod(idx, size)
        for dr_B, dc_B, dr_p1, dc_p1, dr_p2, dc_p2 in BRIDGE_PATTERNS:
            rb,  cb  = r + dr_B,  c + dc_B
            rp1, cp1 = r + dr_p1, c + dc_p1
            rp2, cp2 = r + dr_p2, c + dc_p2
            if not (0 <= rb  < size and 0 <= cb  < size): continue
            if not (0 <= rp1 < size and 0 <= cp1 < size): continue
            if not (0 <= rp2 < size and 0 <= cp2 < size): continue
            bidx = rb * size + cb
            p1   = rp1 * size + cp1
            p2   = rp2 * size + cp2
            if board_arr[bidx] == EMPTY and board_arr[p1] == EMPTY and board_arr[p2] == EMPTY:
                candidates.add(bidx)

    if candidates:
        chosen = random.choice(list(candidates))
        return divmod(chosen, size)

    # Fallback: random empty cell in the middle third of the board
    lo = size // 3
    hi = size - lo          # exclusive upper bound
    middle = [
        idx for idx in board.empty_cells
        if lo <= idx // size < hi and lo <= idx % size < hi
    ]
    pool = middle if middle else board.empty_cells
    return divmod(random.choice(pool), size)


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

    # Count how many stones we have already placed
    stones_placed = size * size - len(board.empty_cells)
    my_stones = (stones_placed + 1) // 2 if my_color == RED else stones_placed // 2
    step = my_stones + 1   # the move we are about to make (1-indexed)

    # Steps 1-5: bridge-forming phase
    if step <= size // 2 :
        return bridge_move_random(board, my_color)

    # Steps 6-15: cut the opponent's shortest path
    if step <= 30:
        cuts = cut_cells(board, my_color)
        if cuts:
            return divmod(random.choice(cuts), size)

    # Steps 16+: full MCTS
    return mcts_search(board, my_color, time_limit)


# ---------------------------------------------------------------------------
# Time budget per board size
# ---------------------------------------------------------------------------

# leave 50ms for I/O and overhead.
_TIME_BUDGET = {
    11: 0.10,   # limit 150 ms → use 120 ms
    15: 0.15,   # limit 200 ms → use 160 ms
    19: 0.20,   # limit 250 ms → use 200 ms
    21: 0.25,   # limit 300 ms → use 240 ms
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
        
        # line = '11 BLUE 0:5:B,1:7:B,1:9:B,2:0:B,2:2:R,2:9:B,3:2:R,3:5:R,3:7:R,3:8:B,3:9:R,4:3:R,4:5:R,4:6:R,4:7:R,4:8:R,5:4:R,5:5:B,5:6:R,5:8:R,6:1:R,6:3:R,6:4:R,6:9:B,7:2:R,7:9:B,8:3:B,8:4:B,8:5:B,9:3:B,9:8:B,10:5:B,10:10:B'
        # print(line, file=sys.stderr)
        # size, my_color, board = parse_input(line)
        # time_limit = _TIME_BUDGET.get(size, _DEFAULT_TIME)
        # result = choose_move(size, board, my_color, time_limit)
        # print('swap' if result == 'swap' else f"{result[0]} {result[1]}")
        # sys.stdout.flush()
        gc.disable()                      # stop GC during our turn
        try:
            size, my_color, board = parse_input(line)
            time_limit = _TIME_BUDGET.get(size, _DEFAULT_TIME)
            result = choose_move(size, board, my_color, time_limit)
            print('swap' if result == 'swap' else f"{result[0]} {result[1]}")
            sys.stdout.flush()
        finally:
            gc.enable()
        gc.collect()                      # GC runs NOW, after move is sent

if __name__ == '__main__':
    main()
