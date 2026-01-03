"""Quick visualization of life token loss over a Hanabi game log.

Usage:
    python scripts/visualize_lives.py logs/2025_12_17_PROB_LOG_AGENT2.log
    python scripts/visualize_lives.py logs --out lives.png
    python scripts/visualize_lives.py logs_probabilistic
"""

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


Update = Tuple[int, int, int, str]  # (step, lives, clues, timestamp)


def parse_log(log_path: Path) -> Dict[int, List[Update]]:
    """Parse tracker updates from the log."""
    episode_data: Dict[int, List[Update]] = {}
    header_pattern = re.compile(r"^\[(?P<ts>[^]]+)\]\s+\[EPISODE_(?P<ep>\d+)\]")
    tracker_pattern = re.compile(r"Game state tracker updated:\s+(?P<payload>\{.*\})")

    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        header_match = header_pattern.match(line)

        if header_match:
            ts = header_match.group("ts")
            ep = int(header_match.group("ep"))

            # Look ahead in the next few lines for tracker update
            for j in range(i, min(i + 5, len(lines))):
                tracker_match = tracker_pattern.search(lines[j])
                if tracker_match:
                    payload_text = tracker_match.group("payload")
                    try:
                        payload = ast.literal_eval(payload_text)
                        lives = payload.get("lives")
                        clues = payload.get("clues")

                        if lives is not None and clues is not None:
                            episode_data.setdefault(ep, [])
                            episode_data[ep].append((len(episode_data[ep]) + 1, lives, clues, ts))
                    except Exception:
                        # Skip malformed lines without stopping the script
                        pass
                    break

        i += 1

    return episode_data


def plot_episode(ep: int, updates: List[Update], out_path: Path):
    """Plot lives over time for a single episode."""
    steps = [u[0] for u in updates]
    lives = [u[1] for u in updates]

    plt.figure(figsize=(8, 4))
    plt.plot(steps, lives, marker="o", label=f"Episode {ep}")
    plt.yticks([0, 1, 2, 3])
    plt.ylim(-0.2, 3.2)
    plt.xlabel("Update index")
    plt.ylabel("Lives")
    plt.title(f"Life tokens over time (Episode {ep})")

    # Highlight life losses
    for i in range(1, len(updates)):
        prev_lives = updates[i - 1][1]
        curr_lives = updates[i][1]
        if curr_lives < prev_lives:
            step, _, _, ts = updates[i]
            plt.scatter(step, curr_lives, color="red", zorder=3)
            plt.text(step, curr_lives - 0.1, f"-{prev_lives - curr_lives} @ {ts}", ha="center", va="top", fontsize=8, rotation=45)

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def find_log_files(root: Path) -> List[Path]:
    """Find agent log files under a directory."""
    if root.is_file():
        return [root]

    candidates = sorted(root.glob("agent_*.log"))
    if candidates:
        return candidates

    return sorted(root.glob("*.log"))


def main():
    parser = argparse.ArgumentParser(description="Visualize lives lost from a Hanabi log.")
    parser.add_argument("logfile", type=Path, help="Path to the log file or directory")
    parser.add_argument("--out", type=Path, help="Output image path or directory")
    args = parser.parse_args()

    log_files = find_log_files(args.logfile)
    if not log_files:
        raise SystemExit("No log files found at the given path.")

    output_is_dir = args.out is not None and (args.out.is_dir() or args.out.suffix == "")
    if args.out and output_is_dir:
        args.out.mkdir(parents=True, exist_ok=True)

    for log_file in log_files:
        data = parse_log(log_file)
        if not data:
            print(f"Skipping {log_file} (no tracker updates found)")
            continue

        for ep, updates in data.items():
            if not updates:
                continue

            if args.out:
                if output_is_dir:
                    out_path = args.out / f"{log_file.stem}.ep{ep}.lives.png"
                else:
                    out_path = args.out
            else:
                out_path = log_file.with_suffix(".lives.png")

            if len(data) > 1:
                out_path = out_path.with_stem(out_path.stem + f".ep{ep}")

            plot_episode(ep, updates, out_path)
            print(f"Saved Episode {ep} plot to {out_path}")


if __name__ == "__main__":
    main()
