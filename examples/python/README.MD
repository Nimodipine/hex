# Hex Agent

**Authors:** Wei-Yun Feng, Luochun Xiao

This agent is designed to play the game of Hex on various board sizes using a Monte Carlo Tree Search (MCTS) architecture. It is optimized for competitive play with strict time constraints.

---

## How to Run

Play as RED (human) vs the agent as BLUE:
```bash
python3 gui_main.py --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Play as BLUE (human) vs the agent as RED:
```bash
python3 gui_main.py --red-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Agent vs. Agent:
```bash
python3 gui_main.py \
  --red-subprocess "python3 ./examples/python/mcts_random_agent.py" \
  --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Custom board size (e.g. 19):
```bash
python3 gui_main.py --board-size 19 --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Custom timeout per move (e.g. 0.5 seconds):
```bash
python3 gui_main.py --timeout 0.5 --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Custom memory limit (e.g. 128 MB):
```bash
python3 gui_main.py --memory-limit 128 --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
```

Combined example — 19×19 board, 0.3s timeout, 128 MB memory, agent vs. agent:
```bash
python3 gui_main.py \
  --board-size 19 \
  --timeout 0.3 \
  --memory-limit 128 \
  --red-subprocess "python3 ./examples/python/mcts_random_agent.py" \
  --blue-subprocess "python3 ./examples/python/mcts_random_agent.py"
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
| `random` | Random move sampling during rollouts |
| `time` | Deadline enforcement across MCTS phases |
| `gc` | Manual garbage collection control |
| `collections.deque` | Double-ended queue for 0-1 BFS |
| `array` | Compact typed arrays for fast `UnionFind` copying |

**Python version:** 3.8 or higher recommended.

---

## What is implemented

1. **Core MCTS (UCT):** builds a search tree within the time limit and picks the most-visited move, including selection, expansion, simulation, and backpropagation.

2. **Threat-aware rollout:** instead of pure random cell assignment, the simulation phase detects opponent "threat cells" (empty cells adjacent to 2+ opponent stones) and blocks them before falling back to random play. This produces more realistic simulations and gives MCTS stronger signal.

3. **Heuristic move ordering:** candidate moves are scored before MCTS expansion using a gain-based BFS formula rather than being shuffled randomly. Moves are ranked by:
   - `me_gain × 2.0` — how much the move shortens my own shortest path
   - `opp_loss × block_weight` — how much it lengthens the opponent's path, with `block_weight` scaling from 1.2 to 4.0 as the opponent gets closer to winning (urgency scaling)
   - `bottleneck_bonus` — extra weight for cells that lie on many opponent near-optimal paths, catching diagonal chains that single-path BFS would miss

4. **Bidirectional 0-1 BFS (`fast_bfs`):** threat detection runs BFS from both edges of the board and returns the minimum, making urgency scoring symmetric regardless of which end of the board the opponent builds from.

5. **Bottleneck detection (`_count_shortest_path_uses`):** runs forward + backward BFS to identify empty cells used by many opponent near-optimal paths. Addresses the key weakness of single-path BFS where blocking one cell of a diagonal chain doesn't change the minimum distance.

6. **Neighbor pruning:** candidate moves are restricted to cells adjacent to existing stones, reducing the search space significantly. Falls back to all empty cells only when the board is empty or no neighbors exist. Capped at 20 candidates per node with random sampling if more exist.

7. **Precomputed neighbor table (`_get_neighbor_table`):** per-cell neighbor index lists are built once per board size and cached. Eliminates `divmod` and bounds checking inside the BFS hot loop.

8. **Fast `UnionFind` copy:** internal arrays switched from Python `list` to `array.array` (typed compact arrays), reducing `board.copy()` time from ~15ms to ~2µs.

9. **Board class with Union-Find win detection:** incremental connectivity tracking via two Union-Find structures (one per color) with virtual edge nodes.

10. **Opening heuristics (`choose_move`):**
    - RED always takes the center on move 1
    - BLUE swaps if RED takes center, otherwise takes center
    - MCTS for all other turns

11. **Manual garbage collection:** GC is disabled during move calculation and triggered immediately after the move is sent, keeping forfeit rates low.

---
