# Hex Agent

This agent is designed to play the game of Hex on various board sizes using a Monte Carlo Tree Search (MCTS) architecture. It is optimized for competitive play with strict time constraints.

---

## How to Run

Play as RED (human) vs the agent as BLUE:
```bash
python3 gui_main.py --blue-subprocess "python3 ./examples/python/hex_agent.py"
```

Play as BLUE (human) vs the agent as RED:
```bash
python3 gui_main.py --red-subprocess "python3 ./examples/python/hex_agent.py"
```

Agent vs. Agent:
```bash
python3 gui_main.py \
  --red-subprocess "python3 ./examples/python/hex_agent.py" \
  --blue-subprocess "python3 ./examples/python/hex_agent.py"
```

Custom board size (e.g. 19):
```bash
python3 gui_main.py --board-size 19 --blue-subprocess "python3 ./examples/python/hex_agent.py"
```

---

## Current Features

- **MCTS with UCB1 selection** — builds a search tree within the time limit and picks the most-visited move
- **Opening strategy** — RED always claims the center on move 1; BLUE swaps if RED takes the center, otherwise steals it
- **Dynamic board size support** — correctly handles 11×11, 15×15, 19×19, and 21×21
- **Neighbor-only candidate pruning** — on large boards, only considers moves adjacent to existing stones to avoid timeouts
- **Win detection** — BFS-based check used during tree search to identify winning positions

---

## What is Incomplete

- **No rollout simulation** — MCTS evaluates positions immediately after expansion instead of playing out random games to a terminal state, which weakens the search significantly
- **Weak endgame play** — without rollouts or a heuristic, the agent struggles in the midgame and endgame when the opening advantage runs out
- **No explicit defense logic** — while MCTS naturally blocks threats it sees in simulations, there is no dedicated bridge-breaking logic, which leaves gaps against high-level human players
- **No state persistence** — the search tree is fully rebuilt every turn; carrying the tree over from the opponent's turn would effectively double the simulation count at no extra time cost