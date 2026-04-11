#!/usr/bin/env python3
import sys
import time
import math
import random
from collections import deque

def parse_board(line):
    parts = line.strip().split(maxsplit=2)
    if len(parts) < 2: return 11, "RED", {}
    size = int(parts[0])
    my_color = parts[1]
    board = {}
    if len(parts) == 3 and parts[2]:
        for move in parts[2].split(','):
            row, col, color = move.split(':')
            board[(int(row), int(col))] = color
    return size, my_color, board

def get_neighbors(r, c, size):
    for dr, dc in [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc

def check_win(board, size, color):
    target = 'R' if color == 'RED' else 'B'
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
    
    # Candidate pruning for larger boards (19x19, 21x21)
    placed = list(board.keys())
    candidates = set()
    for (r, c) in placed:
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in board: candidates.add((nr, nc))
    
    root.untried_moves = list(candidates) if candidates else [(size // 2, size // 2)]

    while time.time() - start_time < time_limit:
        node = root
        temp_board = board.copy()
        
        # 1. Selection
        while not node.untried_moves and node.children:
            node = max(node.children, key=lambda c: c.ucb1())
            temp_board[node.move] = my_p if node.parent == root else opp_p

        # 2. Expansion
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            temp_board[move] = my_p if node == root else opp_p
            new_node = MCTSNode(move=move, parent=node)
            node.children.append(new_node)
            node = new_node

        # 3. Fast Simulation (limit steps for 21x21 efficiency)
        won = check_win(temp_board, size, my_color)
        
        # 4. Backpropagation
        while node:
            node.visits += 1
            if won: node.wins += 1
            node = node.parent
            
    return max(root.children, key=lambda c: c.visits).move if root.children else (size // 2, size // 2)

def choose_move(size, my_color, board):
    # DYNAMIC CENTER CALCULATION
    center = size // 2
    
    # 1. As RED: Always take center regardless of size
    if not board and my_color == 'RED':
        return (center, center)

    # 2. As BLUE: Dynamic Swap Logic
    if my_color == 'BLUE' and len(board) == 1:
        (r, c), _ = next(iter(board.items()))
        # If RED took the center of this specific board size, SWAP
        if r == center and c == center:
            return 'swap'
        # If RED missed the center, BLUE takes it
        elif (center, center) not in board:
            return (center, center)

    return mcts_search(size, my_color, board)

def main():
    while True:
        try:
            line = sys.stdin.readline()
            if not line: break
            size, my_color, board = parse_board(line)
            result = choose_move(size, my_color, board)
            print('swap' if result == 'swap' else f"{result[0]} {result[1]}")
            sys.stdout.flush()
        except EOFError: break

if __name__ == "__main__":
    main()