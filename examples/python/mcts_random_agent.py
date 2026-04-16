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

# Pre-computed per-cell neighbor index lists keyed by board size.
# Built once on first use and cached — eliminates divmod + bounds checks
# inside the hot BFS loop.
_NEIGHBOR_TABLE_CACHE = {}

def _get_neighbor_table(size):
    if size not in _NEIGHBOR_TABLE_CACHE:
        table = []
        for idx in range(size * size):
            r, c = divmod(idx, size)
            nbrs = []
            for dr, dc in NEIGHBORS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    nbrs.append(nr * size + nc)
            table.append(nbrs)
        _NEIGHBOR_TABLE_CACHE[size] = table
    return _NEIGHBOR_TABLE_CACHE[size]


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------

class UnionFind:
    __slots__ = ['parent', 'rank']

    def __init__(self, n):
        # Use array module for compact, fast-copying storage
        import array
        self.parent = array.array('i', range(n))
        self.rank   = array.array('b', [0] * n)

    def find(self, x):
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

    def copy(self):
        import array
        uf = UnionFind.__new__(UnionFind)
        uf.parent = array.array('i', self.parent)
        uf.rank   = array.array('b', self.rank)
        return uf


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
        new = HexBoard.__new__(HexBoard)
        new.size        = self.size
        new.board       = self.board[:]
        new.uf_red      = self.uf_red.copy()
        new.uf_blue     = self.uf_blue.copy()
        new.empty_cells = self.empty_cells[:]
        new.RED_TOP     = self.RED_TOP
        new.RED_BOTTOM  = self.RED_BOTTOM
        new.BLUE_LEFT   = self.BLUE_LEFT
        new.BLUE_RIGHT  = self.BLUE_RIGHT
        return new


# ---------------------------------------------------------------------------
# Fast rollout (no board copy needed)
# ---------------------------------------------------------------------------

def _find_threat(assignment, empty_set, size, attacker):
    best = -1
    best_score = -1
    for idx in empty_set:
        row, col = divmod(idx, size)
        score = 0
        
        # 1. Neighbor Count (The core threat logic)
        neighbor_count = 0
        for dr, dc in NEIGHBORS:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                if assignment[nr * size + nc] == attacker:
                    neighbor_count += 1
        
        # 2. Edge Proximity (Prioritize blocking the attacker's goal edges)
        edge_bonus = 0
        if attacker == RED: # Red wants row 0 and row size-1
            if row == 0 or row == size - 1:
                edge_bonus = 2
        else: # Blue wants col 0 and col size-1
            if col == 0 or col == size - 1:
                edge_bonus = 2

        # Combine scores: prioritize neighbors, then edges
        total_score = (neighbor_count * 3) + edge_bonus
        
        if total_score > best_score:
            best_score = total_score
            best = idx
            
    return best if best_score >= 3 else -1 # Only return if it's a real threat


def rollout(board, current_player, deadline):
    """
    Simulate a game to completion using a light threat-aware policy:
      - Before placing randomly, check whether the OPPONENT has a "threat cell"
        (an empty cell touching 2+ of their stones).  If so, block it.
      - Otherwise place randomly.

    This makes the agent actually respond to chains being built rather than
    ignoring them with purely random play.

    Returns RED, BLUE, or None if the deadline was exceeded mid-rollout.
    """
    size = board.size
    # Work on mutable copies — no board mutation
    assignment = board.board[:]
    remaining  = board.empty_cells[:]
    random.shuffle(remaining)          # shuffle for random fallback order

    # Use a set for O(1) membership checks and removal
    empty_set = set(remaining)

    player = current_player
    step   = 0
    while empty_set:
        # Check deadline every 64 steps
        if deadline is not None and step % 64 == 0 and time.time() >= deadline:
            return None

        opponent = BLUE if player == RED else RED

        # Look for a cell the opponent threatens — block it
        block = _find_threat(assignment, empty_set, size, opponent)
        if block != -1:
            chosen = block
        else:
            # No urgent threat: pick the next pre-shuffled random cell
            # Pop from the end of remaining; skip cells already taken
            chosen = -1
            while remaining:
                c = remaining.pop()
                if c in empty_set:
                    chosen = c
                    break
            if chosen == -1:
                break   # empty_set non-empty but remaining exhausted (shouldn't happen)

        assignment[chosen] = player
        empty_set.discard(chosen)
        player = opponent
        step += 1

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
        if deadline is not None and head % 32 == 0 and time.time() >= deadline:
            return None

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
        self.board = board
        self.parent = parent
        self.move = move
        self.current_player = current_player
        self.children = []
        self.visits = 0
        self.wins = 0.0

        # 1. Prune candidates to neighbors only
        candidates = self.get_pruned_candidates(board)
        
        # 2. Score moves: How much does this move help the opponent?
        opp_color = BLUE if current_player == RED else RED

        if len(candidates) > 20:
            random.shuffle(candidates)
            candidates = candidates[:20]

        # Pre-compute baselines and bottleneck map once for all candidates
        base_me  = fast_bfs(board.board, board.size, self.current_player)
        base_opp = fast_bfs(board.board, board.size, opp_color)
        if base_opp <= board.size // 2:
            opp_uses = _count_shortest_path_uses(board.board, board.size, opp_color, budget=1)
        else:
            opp_uses = {}

        scored_moves = []
        for m_idx in candidates:
            val = get_combined_score(board.board, board.size, self.current_player, m_idx,
                                     opp_uses, base_me, base_opp)
            scored_moves.append((val, m_idx))

        # Sort so that the HIGHEST combined_value is at the end for .pop()
        scored_moves.sort(key=lambda x: x[0]) 
        self.untried_moves = [m[1] for m in scored_moves]
    
    def get_pruned_candidates(self, board):
        """
        Returns empty cells adjacent to existing stones, plus any critical
        blocking cells when the opponent is 1-2 moves from winning.
        """
        size = board.size
        if len(board.empty_cells) == size * size:
            return board.empty_cells[:]

        candidates = set()
        for idx, color in enumerate(board.board):
            if color != EMPTY:
                r, c = divmod(idx, size)
                for dr, dc in NEIGHBORS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        nidx = nr * size + nc
                        if board.board[nidx] == EMPTY:
                            candidates.add(nidx)

        return list(candidates) if candidates else board.empty_cells[:]


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
    
_BFS_INF = 9999

def _bfs_one_direction(assignment, size, color, start_indices, goal_val, goal_is_row):
    """
    0-1 BFS from a set of seed cells toward a goal edge.
    - Flat-array `dist` replaces dict visited  → no hash overhead
    - Stores only the index in the deque        → no tuple allocation
    - Uses precomputed neighbor table           → no divmod in hot loop
    """
    opp_color = BLUE if color == RED else RED
    nbr_table = _get_neighbor_table(size)
    dist = [_BFS_INF] * (size * size)
    queue = deque()

    for idx in start_indices:
        if assignment[idx] == opp_color:
            continue
        cost = 0 if assignment[idx] == color else 1
        if cost < dist[idx]:
            dist[idx] = cost
            if cost == 0:
                queue.appendleft(idx)
            else:
                queue.append(idx)

    if not queue:
        return _BFS_INF

    if goal_is_row:
        while queue:
            idx = queue.popleft()
            d = dist[idx]
            if idx // size == goal_val:
                return d
            for nidx in nbr_table[idx]:
                if assignment[nidx] == opp_color:
                    continue
                cost = 0 if assignment[nidx] == color else 1
                nd = d + cost
                if nd < dist[nidx]:
                    dist[nidx] = nd
                    if cost == 0:
                        queue.appendleft(nidx)
                    else:
                        queue.append(nidx)
    else:
        while queue:
            idx = queue.popleft()
            d = dist[idx]
            if idx % size == goal_val:
                return d
            for nidx in nbr_table[idx]:
                if assignment[nidx] == opp_color:
                    continue
                cost = 0 if assignment[nidx] == color else 1
                nd = d + cost
                if nd < dist[nidx]:
                    dist[nidx] = nd
                    if cost == 0:
                        queue.appendleft(nidx)
                    else:
                        queue.append(nidx)

    return _BFS_INF


def fast_bfs(assignment, size, color):
    """
    0-1 BFS returning shortest path distance for `color`.
    Runs from BOTH edges and returns the minimum for directional symmetry.
    """
    if color == RED:
        top_starts    = [c for c in range(size) if assignment[c] != BLUE]
        bottom_starts = [c + (size-1)*size for c in range(size) if assignment[c+(size-1)*size] != BLUE]
        d1 = _bfs_one_direction(assignment, size, color, top_starts,    size-1, True)
        d2 = _bfs_one_direction(assignment, size, color, bottom_starts, 0,      True)
    else:
        left_starts  = [r*size       for r in range(size) if assignment[r*size]       != RED]
        right_starts = [r*size+size-1 for r in range(size) if assignment[r*size+size-1] != RED]
        d1 = _bfs_one_direction(assignment, size, color, left_starts,  size-1, False)
        d2 = _bfs_one_direction(assignment, size, color, right_starts, 0,      False)
    return min(d1, d2)


def _count_shortest_path_uses(board_list, size, color, budget):
    """
    Count how many times each empty cell appears on a shortest path for `color`,
    by running a forward BFS (distances from start edge) and backward BFS
    (distances from goal edge) and keeping cells where:
        fwd[cell] + bwd[cell] == shortest_path_length
    and fwd[cell] + bwd[cell] <= budget.

    Returns a dict {idx: count} where count >= 1 means the cell is on
    at least one near-optimal path.
    """
    opp_color = BLUE if color == RED else RED

    def bfs_distances(start_indices):
        nbr_table = _get_neighbor_table(size)
        dist = [_BFS_INF] * (size * size)
        queue = deque()
        for idx in start_indices:
            if board_list[idx] == opp_color:
                continue
            cost = 0 if board_list[idx] == color else 1
            if cost < dist[idx]:
                dist[idx] = cost
                if cost == 0:
                    queue.appendleft(idx)
                else:
                    queue.append(idx)
        while queue:
            idx = queue.popleft()
            d = dist[idx]
            for nidx in nbr_table[idx]:
                if board_list[nidx] == opp_color:
                    continue
                cost = 0 if board_list[nidx] == color else 1
                nd = d + cost
                if nd < dist[nidx]:
                    dist[nidx] = nd
                    if cost == 0:
                        queue.appendleft(nidx)
                    else:
                        queue.append(nidx)
        return dist

    if color == RED:
        fwd_starts  = [c for c in range(size) if board_list[c] != opp_color]
        bwd_starts  = [((size-1)*size + c) for c in range(size) if board_list[(size-1)*size+c] != opp_color]
    else:
        fwd_starts  = [r*size for r in range(size) if board_list[r*size] != opp_color]
        bwd_starts  = [r*size+(size-1) for r in range(size) if board_list[r*size+(size-1)] != opp_color]

    fwd = bfs_distances(fwd_starts)
    bwd = bfs_distances(bwd_starts)

    sp = fast_bfs(board_list, size, color)
    if sp >= size * size:
        return {}

    uses = {}
    for idx in range(size * size):
        if board_list[idx] == EMPTY:
            f = fwd[idx]
            b = bwd[idx]
            if f + b <= sp + budget:
                uses[idx] = uses.get(idx, 0) + 1
    return uses


def get_combined_score(board_list, size, my_color, move_idx, opp_uses=None, base_me=None, base_opp=None):
    """
    Score a candidate move combining:
      1. Path gain: how much MY shortest path shortens
      2. Path loss: how much OPPONENT's shortest path lengthens (with urgency scaling)
      3. Bottleneck bonus: how many near-optimal opponent paths pass through this cell
         (catches diagonal chains where a single block doesn't change min-distance
         but does disrupt many routes at once)
    """
    opp_color = BLUE if my_color == RED else RED

    if base_me  is None: base_me  = fast_bfs(board_list, size, my_color)
    if base_opp is None: base_opp = fast_bfs(board_list, size, opp_color)

    temp = board_list[:]
    temp[move_idx] = my_color
    after_me  = fast_bfs(temp, size, my_color)
    after_opp = fast_bfs(temp, size, opp_color)

    me_gain  = base_me  - after_me
    opp_loss = after_opp - base_opp

    urgency = max(0.0, 1.0 - base_opp / size)
    block_weight = 1.2 + urgency * 2.8

    score = (me_gain * 2.0) + (opp_loss * block_weight)

    # Bottleneck bonus: cells used by many opponent near-optimal paths are
    # high-value blocks even when min-distance doesn't change.
    # opp_uses is pre-computed per node to avoid re-running BFS every call.
    if opp_uses:
        bottleneck_count = opp_uses.get(move_idx, 0)
        if bottleneck_count > 0:
            score += bottleneck_count * (0.5 + urgency * 2.5)

    return score
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