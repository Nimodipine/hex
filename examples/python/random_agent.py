#!/usr/bin/env python3
import sys
import time
import math
import random
from collections import deque

def parse_board(line):
    parts = line.strip().split(maxsplit=2)
    if len(parts) < 2: return 11, "RED", {}
    size, my_color = int(parts[0]), parts[1]
    board = {}
    if len(parts) == 3 and parts[2]:
        for move in parts[2].split(','):
            r, c, clr = move.split(':')
            board[(int(r), int(c))] = clr
    return size, my_color, board

def get_neighbors(r, c, size):
    for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc

def check_win(board, size, color):
    """Fast Union-Find or BFS to check for a winner."""
    target = 'R' if color == 'RED' else 'B'
    opp = 'B' if target == 'R' else 'R'
    if target == 'R':
        starts = [(0, c) for c in range(size) if board.get((0, c)) == 'R']
        is_goal = lambda r, c: r == size - 1
    else:
        starts = [(r, 0) for r in range(size) if board.get((r, 0)) == 'B']
        is_goal = lambda r, c: c == size - 1
    
    if not starts: return False
    queue, visited = deque(starts), set(starts)
    while queue:
        r, c = queue.popleft()
        if is_goal(r, c): return True
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in visited and board.get((nr, nc)) == target:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False

# --- MCTS CORE ---
class MCTSNode:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = None

    def ucb1(self, exploration=1.41):
        if self.visits == 0: return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts_search(size, my_color, board, time_limit=0.13):
    start_time = time.time()
    my_p = 'R' if my_color == 'RED' else 'B'
    opp_p = 'B' if my_p == 'R' else 'R'
    
    root = MCTSNode()
    # Prune candidates: only cells near existing pieces to save time
    placed = list(board.keys())
    if not placed: return (size // 2, size // 2)
    
    candidates = set()
    for (r, c) in placed:
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in board: candidates.add((nr, nc))
    root.untried_moves = list(candidates) if candidates else [(0,0)]

    while time.time() - start_time < time_limit:
        node = root
        state = board.copy()
        
        # 1. Selection
        while node.untried_moves == [] and node.children != []:
            node = max(node.children, key=lambda c: c.ucb1())
            state[node.move] = my_p if node.parent == root else opp_p # Simplified alternate

        # 2. Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            state[move] = my_p if node == root else opp_p
            new_node = MCTSNode(move=move, parent=node)
            node.children.append(new_node)
            node = new_node

        # 3. Simulation (Random Playout)
        curr_p = opp_p # Assume opponent turn after expansion
        sim_board = state.copy()
        empty = [pos for pos in [(r,c) for r in range(size) for c in range(size)] if pos not in sim_board]
        random.shuffle(empty)
        
        while not check_win(sim_board, size, 'RED') and not check_win(sim_board, size, 'BLUE') and empty:
            m = empty.pop()
            sim_board[m] = curr_p
            curr_p = 'R' if curr_p == 'B' else 'B'
        
        # 4. Backpropagation
        won = check_win(sim_board, size, my_color)
        while node is not None:
            node.visits += 1
            if won: node.wins += 1
            node = node.parent
            
    return max(root.children, key=lambda c: c.visits).move

def choose_move(size, my_color, board):
    center = size // 2
    # Standard Opening/Swap Logic
    if not board and my_color == 'RED': return (center, center)
    if my_color == 'BLUE' and len(board) == 1:
        (r, c), _ = next(iter(board.items()))
        if r == center and c == center: return 'swap'
        if (center, center) not in board: return (center, center)

    return mcts_search(size, my_color, board)

def main():
    # Use sys.stdin for faster line-by-line processing
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            size, color, board = parse_board(line)
            result = choose_move(size, color, board)
            if result == 'swap':
                print("swap")
            else:
                print(f"{result[0]} {result[1]}")
            sys.stdout.flush()
        except EOFError:
            break

if __name__ == "__main__":
    main()