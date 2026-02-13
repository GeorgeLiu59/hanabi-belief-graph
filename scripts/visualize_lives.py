"""Enhanced visualization of life token loss over Hanabi game logs.

Features:
    - Single game visualization (lives and/or clues)
    - Overlay mode for multiple games with statistics (mean, std dev, median, mode)
    - Histogram of rounds to first life loss

Usage:
    # Single game
    python scripts/visualize_lives.py logs/bg_probabilistic/agent_*.log

    # Overlay multiple games with statistics
    python scripts/visualize_lives.py logs/bg_probabilistic/ --overlay --out overlay.png

    # Show both lives and clues
    python scripts/visualize_lives.py logs/bg_probabilistic/ --overlay --metric both

    # Histogram of first life loss rounds
    python scripts/visualize_lives.py logs/bg_probabilistic/ --histogram
"""

import argparse
import ast
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


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


def get_first_life_loss_round(updates: List[Update]) -> Optional[int]:
    """Get the round number where the first life was lost."""
    for i in range(1, len(updates)):
        if updates[i][1] < updates[i-1][1]:
            return updates[i][0]
    return None


def compute_mode(values: List[int]) -> Optional[int]:
    """Compute mode of a list of integers."""
    if not values:
        return None
    counter = Counter(values)
    most_common = counter.most_common(1)
    return most_common[0][0] if most_common else None


def plot_episode(ep: int, updates: List[Update], out_path: Path, metric: str = "lives"):
    """Plot lives/clues over time for a single episode."""
    steps = [u[0] for u in updates]
    lives = [u[1] for u in updates]
    clues = [u[2] for u in updates]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if metric in ["lives", "both"]:
        ax1.plot(steps, lives, marker="o", color="red", label="Lives", linewidth=2)
        ax1.set_ylabel("Lives", color="red")
        ax1.set_ylim(-0.2, 3.5)
        ax1.set_yticks([0, 1, 2, 3])

        # Highlight life losses
        for i in range(1, len(updates)):
            if updates[i][1] < updates[i-1][1]:
                ax1.scatter(updates[i][0], updates[i][1], color="darkred", s=100, zorder=5)

    if metric in ["clues", "both"]:
        if metric == "both":
            ax2 = ax1.twinx()
            ax2.plot(steps, clues, marker="s", color="blue", label="Clues", linewidth=2, alpha=0.7)
            ax2.set_ylabel("Clues", color="blue")
            ax2.set_ylim(-0.5, 9)
        else:
            ax1.plot(steps, clues, marker="s", color="blue", label="Clues", linewidth=2)
            ax1.set_ylabel("Clues", color="blue")
            ax1.set_ylim(-0.5, 9)

    ax1.set_xlabel("Turn")
    ax1.set_title(f"Episode {ep} - Game Progression")
    ax1.grid(alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    if metric == "both":
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_overlay(
    all_games: List[Tuple[Path, Dict[int, List[Update]]]],
    out_path: Path,
    metric: str = "lives"
):
    """Plot overlay of multiple games with statistical measures."""
    max_turns = 0
    lives_series = []
    clues_series = []
    first_life_loss_rounds = []

    for log_path, episode_data in all_games:
        for ep, updates in episode_data.items():
            if not updates:
                continue
            turns = [u[0] for u in updates]
            lives = [u[1] for u in updates]
            clues = [u[2] for u in updates]
            max_turns = max(max_turns, max(turns))
            lives_series.append((turns, lives))
            clues_series.append((turns, clues))
            first_loss = get_first_life_loss_round(updates)
            if first_loss is not None:
                first_life_loss_rounds.append(first_loss)

    if not lives_series:
        print("No data to plot")
        return

    # Create aligned arrays for statistics
    lives_matrix = np.full((len(lives_series), max_turns), np.nan)
    clues_matrix = np.full((len(clues_series), max_turns), np.nan)

    for i, (turns, lives) in enumerate(lives_series):
        for t, l in zip(turns, lives):
            lives_matrix[i, t-1] = l

    for i, (turns, clues) in enumerate(clues_series):
        for t, c in zip(turns, clues):
            clues_matrix[i, t-1] = c

    turn_range = np.arange(1, max_turns + 1)
    lives_mean = np.nanmean(lives_matrix, axis=0)
    lives_std = np.nanstd(lives_matrix, axis=0)
    lives_median = np.nanmedian(lives_matrix, axis=0)
    clues_mean = np.nanmean(clues_matrix, axis=0)
    clues_std = np.nanstd(clues_matrix, axis=0)
    clues_median = np.nanmedian(clues_matrix, axis=0)

    if metric == "both":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = None

    # Plot lives
    if metric in ["lives", "both"]:
        ax = ax1
        for turns, lives in lives_series:
            ax.plot(turns, lives, color="red", alpha=0.1, linewidth=0.5)
        ax.plot(turn_range, lives_mean, color="red", linewidth=2.5, label="Mean")
        ax.fill_between(turn_range, lives_mean - lives_std, lives_mean + lives_std,
                        color="red", alpha=0.2, label="Std Dev")
        ax.plot(turn_range, lives_median, color="darkred", linewidth=2, linestyle="--", label="Median")
        ax.set_ylabel("Lives", fontsize=12)
        ax.set_ylim(-0.2, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(f"Lives Over Time (n={len(lives_series)} games)", fontsize=14)

    # Plot clues
    if metric in ["clues", "both"]:
        ax = ax2 if metric == "both" else ax1
        for turns, clues in clues_series:
            ax.plot(turns, clues, color="blue", alpha=0.1, linewidth=0.5)
        ax.plot(turn_range, clues_mean, color="blue", linewidth=2.5, label="Mean")
        ax.fill_between(turn_range, clues_mean - clues_std, clues_mean + clues_std,
                        color="blue", alpha=0.2, label="Std Dev")
        ax.plot(turn_range, clues_median, color="darkblue", linewidth=2, linestyle="--", label="Median")
        ax.set_ylabel("Clues", fontsize=12)
        ax.set_ylim(-0.5, 9)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        ax.set_title(f"Clues Over Time (n={len(clues_series)} games)", fontsize=14)

    ax_bottom = ax2 if metric == "both" else ax1
    ax_bottom.set_xlabel("Turn", fontsize=12)

    # Stats text box
    stats_text = f"Games: {len(lives_series)}"
    if first_life_loss_rounds:
        flr_mean = mean(first_life_loss_rounds)
        flr_median = median(first_life_loss_rounds)
        flr_mode = compute_mode(first_life_loss_rounds)
        flr_std = stdev(first_life_loss_rounds) if len(first_life_loss_rounds) > 1 else 0
        stats_text += f"\n\nFirst Life Loss Round:"
        stats_text += f"\n  Mean: {flr_mean:.1f}"
        stats_text += f"\n  Median: {flr_median:.1f}"
        stats_text += f"\n  Mode: {flr_mode}"
        stats_text += f"\n  Std Dev: {flr_std:.1f}"
        stats_text += f"\n  Games with loss: {len(first_life_loss_rounds)}/{len(lives_series)}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
             bbox=props, family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Overlay plot saved to {out_path}")


def plot_histogram(
    all_games: List[Tuple[Path, Dict[int, List[Update]]]],
    out_path: Path
):
    """Plot histogram of rounds to first life loss with statistics."""
    first_life_loss_rounds = []
    for log_path, episode_data in all_games:
        for ep, updates in episode_data.items():
            first_loss = get_first_life_loss_round(updates)
            if first_loss is not None:
                first_life_loss_rounds.append(first_loss)

    if not first_life_loss_rounds:
        print("No life losses found")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = range(0, max(first_life_loss_rounds) + 5, 2)
    ax.hist(first_life_loss_rounds, bins=bins, color='red', alpha=0.7, edgecolor='black')

    flr_mean = mean(first_life_loss_rounds)
    flr_median = median(first_life_loss_rounds)
    flr_mode = compute_mode(first_life_loss_rounds)
    flr_std = stdev(first_life_loss_rounds) if len(first_life_loss_rounds) > 1 else 0

    ax.axvline(flr_mean, color='blue', linestyle='-', linewidth=2, label=f'Mean: {flr_mean:.1f}')
    ax.axvline(flr_median, color='green', linestyle='--', linewidth=2, label=f'Median: {flr_median:.1f}')
    ax.axvline(flr_mode, color='orange', linestyle=':', linewidth=2, label=f'Mode: {flr_mode}')
    ax.axvspan(flr_mean - flr_std, flr_mean + flr_std, alpha=0.2, color='blue', label=f'Std Dev: {flr_std:.1f}')

    ax.set_xlabel("Round of First Life Loss", fontsize=12)
    ax.set_ylabel("Number of Games", fontsize=12)
    ax.set_title(f"Distribution of First Life Loss (n={len(first_life_loss_rounds)} games)", fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Histogram saved to {out_path}")


def parse_jsonl_game_state(jsonl_path: Path) -> Dict[int, List[Update]]:
    """Parse game state from JSONL files (new format)."""
    import json
    episode_data: Dict[int, List[Update]] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Skip non-turn entries (like EPISODE_START/END)
                if "turn" not in data or "life_tokens" not in data:
                    continue
                ep = data.get("episode", 0)
                turn = data["turn"]
                lives = data["life_tokens"]
                clues = data.get("information_tokens", 8)
                ts = data.get("timestamp", "")
                episode_data.setdefault(ep, [])
                episode_data[ep].append((turn, lives, clues, ts))
            except (json.JSONDecodeError, KeyError):
                continue

    return episode_data


def find_log_files(root: Path) -> List[Path]:
    """Find agent log files under a directory.

    Supports both old format (agent_*.log) and new format:
    - logs/bg_probabilistic/*.log
    - logs/gemini_base/*.log
    - logs/bg_certainty/*.log
    - logs/bg_theory_of_mind/*.log
    - logs/game_state/*.jsonl (new JSONL format)
    """
    if root.is_file():
        return [root]

    candidates = []

    # Try old format first (for backwards compatibility)
    old_format = sorted(root.glob("agent_*.log"))
    if old_format:
        candidates.extend(old_format)

    # Try new format with subdirectories
    subdirs = ["bg_probabilistic", "bg_certainty", "bg_theory_of_mind", "gemini_base"]
    for subdir in subdirs:
        subdir_path = root / subdir
        if subdir_path.exists() and subdir_path.is_dir():
            candidates.extend(sorted(subdir_path.glob("*.log")))

    # Try JSONL game_state files (new format)
    jsonl_files = sorted(root.glob("*.jsonl"))
    if jsonl_files:
        candidates.extend(jsonl_files)

    # Also check subdirectories for JSONL files
    for subdir in root.iterdir():
        if subdir.is_dir():
            candidates.extend(sorted(subdir.glob("*.jsonl")))

    if candidates:
        return sorted(set(candidates))  # Remove duplicates and sort

    # Fallback: any .log files in root
    return sorted(root.glob("*.log"))


def main():
    parser = argparse.ArgumentParser(description="Enhanced Hanabi game visualization with statistics.")
    parser.add_argument("logfile", type=Path, help="Path to the log file or directory")
    parser.add_argument("--out", type=Path, help="Output image path or directory")
    parser.add_argument("--overlay", action="store_true", help="Create overlay plot of all games with statistics")
    parser.add_argument("--histogram", action="store_true", help="Create histogram of first life loss rounds")
    parser.add_argument("--metric", choices=["lives", "clues", "both"], default="lives",
                       help="Which metric to plot (default: lives)")
    args = parser.parse_args()

    log_files = find_log_files(args.logfile)
    if not log_files:
        raise SystemExit("No log files found at the given path.")

    print(f"Found {len(log_files)} log files")

    # Parse all logs
    all_games: List[Tuple[Path, Dict[int, List[Update]]]] = []
    for log_file in log_files:
        if log_file.suffix == ".jsonl":
            data = parse_jsonl_game_state(log_file)
        else:
            data = parse_log(log_file)
        if data:
            all_games.append((log_file, data))

    print(f"Parsed {len(all_games)} games with data")

    if not all_games:
        raise SystemExit("No valid game data found")

    # Determine output directory
    if args.out:
        out_base = args.out
    else:
        out_base = args.logfile if args.logfile.is_dir() else args.logfile.parent

    if args.overlay:
        # Create overlay plot
        if out_base.suffix:
            out_path = out_base
        else:
            out_base.mkdir(parents=True, exist_ok=True)
            out_path = out_base / f"overlay_{args.metric}.png"
        plot_overlay(all_games, out_path, args.metric)

    if args.histogram:
        # Create histogram
        if out_base.suffix:
            hist_path = out_base.with_stem(out_base.stem + "_histogram")
        else:
            out_base.mkdir(parents=True, exist_ok=True)
            hist_path = out_base / "first_life_loss_histogram.png"
        plot_histogram(all_games, hist_path)

    if not args.overlay and not args.histogram:
        # Plot individual episodes (original behavior)
        output_is_dir = args.out is not None and (args.out.is_dir() or args.out.suffix == "")
        if args.out and output_is_dir:
            args.out.mkdir(parents=True, exist_ok=True)

        for log_path, episode_data in all_games:
            for ep, updates in episode_data.items():
                if not updates:
                    continue

                if args.out:
                    if output_is_dir:
                        out_path = args.out / f"{log_path.stem}.ep{ep}.{args.metric}.png"
                    else:
                        out_path = args.out
                else:
                    out_path = log_path.with_suffix(f".{args.metric}.png")

                if len(episode_data) > 1:
                    out_path = out_path.with_stem(out_path.stem + f".ep{ep}")

                plot_episode(ep, updates, out_path, args.metric)
                print(f"Saved Episode {ep} plot to {out_path}")


if __name__ == "__main__":
    main()
