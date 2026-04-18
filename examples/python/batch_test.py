#!/usr/bin/env python3
"""
Batch test: run two agents against each other N times.
Reports wins, timeouts, and average game length.

Usage:
    python3 batch_test.py --games 100
    python3 batch_test.py --games 200 --board-size 11 --timeout 0.15
"""

import subprocess
import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Batch test two agents")
    parser.add_argument("--games",      type=int,   default=100,  help="Number of games to run")
    parser.add_argument("--board-size", type=int,   default=11,   help="Board size")
    parser.add_argument("--timeout",    type=float, default=0.15, help="Per-move timeout (seconds)")
    parser.add_argument(
        "--red",
        type=str,
        default="python3 examples/python/mcts_random_agent.py",
        help="Red agent command"
    )
    parser.add_argument(
        "--blue",
        type=str,
        default="python3 examples/python/cuts_agent.py",
        help="Blue agent command"
    )
    return parser.parse_args()


def run_game(red_cmd, blue_cmd, board_size, timeout):
    """Run one game via terminal_main.py and return (winner, forfeit, turns, duration)."""
    cmd = [
        sys.executable, "terminal_main.py",
        "--red-subprocess",  red_cmd,
        "--blue-subprocess", blue_cmd,
        "--board-size",      str(board_size),
        "--timeout",         str(timeout),
        "--no-stats",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=board_size * board_size * timeout * 2 + 30,  # generous wall timeout
            cwd="."
        )
        duration = time.time() - start
        stdout = result.stdout
        stderr = result.stderr
    except subprocess.TimeoutExpired:
        return {"winner": "WALL_TIMEOUT", "forfeit": True, "turns": -1, "duration": -1}

    # Parse stdout for winner and turn count
    winner   = "UNKNOWN"
    forfeit  = False
    turns    = 0

    for line in stdout.splitlines():
        if "RED WINS" in line or "Red Player wins" in line or "RED wins" in line:
            winner = "RED"
        elif "BLUE WINS" in line or "Blue Player wins" in line or "BLUE wins" in line:
            winner = "BLUE"
        if "FORFEITED" in line:
            forfeit = True
        if "Turn" in line:
            try:
                turns = max(turns, int(line.split("Turn")[1].split(":")[0].strip()))
            except (ValueError, IndexError):
                pass

    # Also check stderr (framework prints forfeit info there sometimes)
    for line in (stdout + stderr).splitlines():
        if "exceeded timeout" in line or "FORFEITED" in line:
            forfeit = True
        if "RED" in line and ("wins" in line.lower() or "WIN" in line):
            winner = "RED"
        elif "BLUE" in line and ("wins" in line.lower() or "WIN" in line):
            winner = "BLUE"

    return {
        "winner":   winner,
        "forfeit":  forfeit,
        "turns":    turns,
        "duration": duration,
        "stdout":   stdout,
        "stderr":   stderr,
    }


def run_a_round(red, blue, games, board_size, timeout, ):
    print(f"  RED : {red}")
    print(f"  BLUE: {blue}")
    print("-" * 60)

    results = {"RED": 0, "BLUE": 0, "UNKNOWN": 0}
    forfeits = 0
    total_turns = 0
    timeout_games = []

    for i in range(1, games + 1):
        r = run_game(red, blue, board_size, timeout)

        winner  = r["winner"]
        forfeit = r["forfeit"]
        turns   = r["turns"]

        results[winner] = results.get(winner, 0) + 1
        if forfeit:
            forfeits += 1
            timeout_games.append(i)
        if turns > 0:
            total_turns += turns

        status = f"{'FORFEIT ' if forfeit else ''}{winner:5s}"
        print(f"  Game {i:4d}: {status}  turns={turns:3d}  {r['duration']:.2f}s")

        sys.stdout.flush()

    # Summary
    played = games
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Games played : {played}")
    print(f"  RED wins     : {results.get('RED', 0)}  "
          f"({100*results.get('RED',0)/played:.1f}%)")
    print(f"  BLUE wins    : {results.get('BLUE', 0)}  "
          f"({100*results.get('BLUE',0)/played:.1f}%)")
    print(f"  Forfeits     : {forfeits}  ({100*forfeits/played:.1f}%)")
    if total_turns > 0 and played > forfeits:
        print(f"  Avg turns    : {total_turns/(played-forfeits):.1f}")
    if timeout_games:
        print(f"  Forfeit games: {timeout_games}")
    print("=" * 60)

    return (results, forfeits, total_turns, timeout_games)


def main():
    args = parse_args()

    print(f"Batch test: {args.games} games/round, board {args.board_size}×{args.board_size}, "
          f"timeout {args.timeout}s")

    print("\nRound 1")
    results1, forfeits1, turns1, timeouts1 = run_a_round(
        args.red, args.blue, args.games, args.board_size, args.timeout)

    print("\nRound 2  (sides swapped)")
    results2, forfeits2, turns2, timeouts2 = run_a_round(
        args.blue, args.red, args.games, args.board_size, args.timeout)

    # Combined summary across both rounds
    # args.red agent: won as RED in round 1 + won as BLUE in round 2
    # args.blue agent: won as BLUE in round 1 + won as RED in round 2
    total_games       = args.games * 2
    red_agent_wins    = results1.get("RED",  0) + results2.get("BLUE", 0)
    blue_agent_wins   = results1.get("BLUE", 0) + results2.get("RED",  0)
    total_forfeits    = forfeits1 + forfeits2
    total_turns       = turns1 + turns2
    all_timeout_games = timeouts1 + [g + args.games for g in timeouts2]

    print("\n" + "=" * 60)
    print("COMBINED SUMMARY  (both rounds)")
    print(f"  Total games  : {total_games}")
    print(f"  {args.red}")
    print(f"    wins: {red_agent_wins}  ({100*red_agent_wins/total_games:.1f}%)")
    print(f"  {args.blue}")
    print(f"    wins: {blue_agent_wins}  ({100*blue_agent_wins/total_games:.1f}%)")
    print(f"  Forfeits     : {total_forfeits}  ({100*total_forfeits/total_games:.1f}%)")
    if total_turns > 0 and total_games > total_forfeits:
        print(f"  Avg turns    : {total_turns/(total_games-total_forfeits):.1f}")
    if all_timeout_games:
        print(f"  Forfeit games: {all_timeout_games}")
    print("=" * 60)


if __name__ == "__main__":
    main()
