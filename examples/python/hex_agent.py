#!/usr/bin/env python3
import sys
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
    # Generator for memory efficiency and speed
    for dr, dc in [(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            yield nr, nc

def fast_bfs(board, size, color):
    """Ultra-lightweight 0-1 BFS to find shortest path distance."""
    target_p = 'R' if color == 'RED' else 'B'
    opp_p = 'B' if target_p == 'R' else 'R'
    
    if target_p == 'R':
        # Start from top row (row 0)
        queue = deque([(0, c, 0 if board.get((0, c)) == 'R' else 1) 
                       for c in range(size) if board.get((0, c)) != 'B'])
        is_goal = lambda r, c: r == size - 1
    else:
        # Start from left column (col 0)
        queue = deque([(r, 0, 0 if board.get((r, 0)) == 'B' else 1) 
                       for r in range(size) if board.get((r, 0)) != 'R'])
        is_goal = lambda r, c: c == size - 1

    visited = {}
    while queue:
        r, c, d = queue.popleft()
        if (r, c) in visited and visited[(r, c)] <= d: continue
        visited[(r, c)] = d
        if is_goal(r, c): return d
        
        for nr, nc in get_neighbors(r, c, size):
            cell_val = board.get((nr, nc))
            if cell_val == opp_p: continue
            
            cost = 0 if cell_val == target_p else 1
            new_d = d + cost
            if (nr, nc) not in visited or new_d < visited[(nr, nc)]:
                if cost == 0: queue.appendleft((nr, nc, new_d))
                else: queue.append((nr, nc, new_d))
    return 999

def choose_move(size, my_color, board):
    center = size // 2
    my_p = 'R' if my_color == 'RED' else 'B'
    opp_p = 'B' if my_p == 'R' else 'R'
    opp_color = 'BLUE' if my_color == 'RED' else 'RED'

    # --- STRATEGY: CENTER MASTERY ---
    # 1. As RED: Always open with the exact center (5,5)
    if not board and my_color == 'RED':
        return (center, center)

    # 2. As BLUE: Always swap if RED played the center (5,5)
    if my_color == 'BLUE' and len(board) == 1:
        (r, c), _ = next(iter(board.items()))
        if r == center and c == center:
            return 'swap'

    # --- GENERAL PLAY ---
    # 3. Find candidates near existing pieces to save time (0.15s limit)
    placed = list(board.keys())
    candidates = set()
    for (r, c) in placed:
        for nr, nc in get_neighbors(r, c, size):
            if (nr, nc) not in board:
                candidates.add((nr, nc))
    
    # Fallback if board is empty or no neighbors (edge case)
    if not candidates:
        empty = [(r, c) for r in range(size) for c in range(size) if (r, c) not in board]
        return empty[0] if empty else (0,0)

    # 4. Evaluative Scoring (Proactive over Reactive)
    best_move = next(iter(candidates))
    best_score = -9999
    
    # Pre-calculate baselines
    base_me = fast_bfs(board, size, my_color)
    base_opp = fast_bfs(board, size, opp_color)

    for move in candidates:
        board[move] = my_p
        # How much closer does this move get us to winning?
        me_gain = base_me - fast_bfs(board, size, my_color)
        # How much does this move hinder the opponent?
        opp_loss = fast_bfs(board, size, opp_color) - base_opp
        del board[move]
        
        # Scoring: Prioritize own connection (2.0) vs Blocking (1.2)
        score = (me_gain * 2.0) + (opp_loss * 1.2)
        
        if score > best_score:
            best_score = score
            best_move = move
            
    return best_move

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