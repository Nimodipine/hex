# Hex Agent

**Authors:** Wei-Yun Feng, Luochun Xiao

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



## What is implemented

1. Core MCTS (UCT): built a search tree within the time limit and picks the most-visited move, including selection, expansion, simulation and backpropagation. 
2. Simple random rollout: implemented a random cell assignment to simulate the game. Use BFS to check winner.
3. Established the board class and applied union data structure for win detection.
4. Opening Heuristics (`choose_move`)
- RED always takes center on move 1
- BLUE swaps if RED takes center, otherwise takes center
- MCTS for all other turns
5. Adopted manual garbage collect to give more time for the mcts calculation, reaches 1% forfeit rates.

---
## What is incomplete or could be improved

1. Incorporate hex-specific rules in the rollout module
2. Neighbor pruning at the early stage of the game.
3. Reuse the tree between turns to improve efficiency.
4. minimize the forfeit rate
