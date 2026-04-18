# Hex Agent

**Authors:** Wei-Yun Feng, Luochun Xiao

This agent is designed to play the game of Hex on various board sizes using a phased strategy that combines bridge-building heuristics, shortest-path cutting, and Monte Carlo Tree Search (MCTS). It is optimized for competitive play with strict time constraints.

---

## How to Run

Play as RED (human) vs the agent as BLUE:
```bash
python3 gui_main.py --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Play as BLUE (human) vs the agent as RED:
```bash
python3 gui_main.py --red-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Agent vs. Agent:
```bash
python3 gui_main.py \
  --red-subprocess "python3 ./examples/python/mcts_mixed_agent.py" \
  --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Custom board size (e.g. 19):
```bash
python3 gui_main.py --board-size 19 --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Custom timeout per move (e.g. 0.5 seconds):
```bash
python3 gui_main.py --timeout 0.5 --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Custom memory limit (e.g. 128 MB):
```bash
python3 gui_main.py --memory-limit 128 --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

Combined example — 19×19 board, 0.3s timeout, 128 MB memory, agent vs. agent:
```bash
python3 gui_main.py \
  --board-size 19 \
  --timeout 0.3 \
  --memory-limit 128 \
  --red-subprocess "python3 ./examples/python/mcts_mixed_agent.py" \
  --blue-subprocess "python3 ./examples/python/mcts_mixed_agent.py"
```

### Command-Line Options

| Option | Description | Default |
|---|---|---|
| `--board-size N` | Set board size (3–26) | `11` |
| `--timeout SECONDS` | Time limit per move (overrides auto-selection) | Auto (based on board size) |
| `--memory-limit MB` | Memory limit in MB | `64` |
| `--red-subprocess "CMD"` | Command to run RED agent | Human (GUI) |
| `--blue-subprocess "CMD"` | Command to run BLUE agent | Human (GUI) |
| `--red-name "NAME"` | Display name for RED | `"Red Player"` |
| `--blue-name "NAME"` | Display name for BLUE | `"Blue Player"` |

---

## Dependencies

This agent uses only Python standard library modules — no third-party packages are required.

| Module | Purpose |
|---|---|
| `sys` | stdin/stdout communication with the game runner |
| `math` | `log` and `sqrt` for the UCT formula |
| `random` | Random move sampling during rollouts and heuristic phases |
| `time` | Deadline enforcement across MCTS phases |
| `gc` | Manual garbage collection control |
| `collections.deque` | Double-ended queue for 0-1 BFS |

**Python version:** 3.8 or higher recommended.

---

## What is implemented

1. **Phased move selection (`choose_move`):** the agent uses three distinct strategies depending on how many moves it has already placed:
   - **Opening heuristics:** RED always takes the center on move 1. BLUE swaps if RED takes center, otherwise takes center itself.
   - **Early game — bridge-building phase (steps 1 to `size // 2`):** plays `bridge_move_random()`, which finds an empty cell that forms a virtual connection (bridge) with an existing friendly stone. A bridge between anchor A and partner B requires both shared corridor cells to also be empty. Falls back to a random cell in the middle third of the board if no bridges are available.
   - **Mid game — cut phase (steps up to 30):** plays `cut_cells()`, which identifies every empty cell that lies on at least one of the opponent's minimum-resistance paths and picks one at random to block.
   - **Late game (step 30+):** runs full MCTS via `mcts_search()`.

2. **Shortest-path cut detection (`cut_cells`):** runs two passes of 0-1 BFS — one forward from the opponent's start edge, one backward from their goal edge — and marks any empty cell `v` as a cut cell if `dist_fwd[v] + 1 + dist_bwd[v] == shortest`. Placing on a cut cell always forces the opponent to find a new route.

3. **0-1 BFS resistance model (`_bfs_resistance`):** traversal cost is 0 for opponent stones (already placed) and 1 for empty cells (must be filled). Our own stones are impassable (cost = ∞). This models the minimum number of additional stones the opponent needs to complete their path.

4. **Bridge pattern detection (`bridge_move_random`):** six hardcoded offset patterns encode all six bridge orientations on the hex grid. A candidate is valid only if the partner cell and both corridor cells are all empty. Falls back to the center third of the board if no bridge is found.

5. **Core MCTS (UCT):** builds a search tree within the time limit and picks the most-visited child as the final move. Implements selection (UCT), expansion, simulation (pure random rollout), and backpropagation.

6. **Fast random rollout:** remaining empty cells are shuffled and assigned alternately to each player, then a BFS checks RED's connectivity from row 0 to row `size - 1`. No board mutation — the rollout works on a `bytearray` snapshot.

7. **Precomputed rollout tables (`_get_rollout_tables`):** per-cell neighbor index lists and edge seed sets are built once per board size and cached, eliminating repeated `divmod` and bounds checks inside the rollout hot loop.

8. **Board class with Union-Find win detection:** incremental connectivity tracking via two Union-Find structures (one per color) with virtual edge nodes (`RED_TOP`, `RED_BOTTOM`, `BLUE_LEFT`, `BLUE_RIGHT`). Win queries are near-constant time.

9. **Manual garbage collection:** GC is disabled during move calculation and triggered immediately after the move is sent, keeping forfeit rates low under tight time limits.

---

