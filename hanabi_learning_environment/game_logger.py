"""Comprehensive game-level logging for Hanabi sessions.

This module provides logging functionality separate from agent logs:
1. Terminal output capture
2. Game state tracking (fireworks, lives, clues)
3. Event tracking (when lives/clues change)
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, TextIO, Optional


class TerminalLogger:
    """Captures all terminal output to a file while still displaying it."""

    def __init__(self, log_dir: str = "logs/terminal"):
        """Initialize terminal logger.

        Args:
            log_dir: Directory to save terminal logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"{log_dir}/terminal_{timestamp}.log"

        self.file_handle = None
        self.original_stdout = None
        self.original_stderr = None

    def start(self):
        """Start capturing terminal output."""
        self.file_handle = open(self.log_file, 'w', encoding='utf-8')
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        # Use Tee to write to both file and terminal
        sys.stdout = TeeOutput(self.original_stdout, self.file_handle)
        sys.stderr = TeeOutput(self.original_stderr, self.file_handle)

        print(f"[TerminalLogger] Started logging to: {self.log_file}")
        print(f"[TerminalLogger] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def stop(self):
        """Stop capturing terminal output and close file."""
        if self.file_handle:
            print("=" * 80)
            print(f"[TerminalLogger] Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"[TerminalLogger] Log saved to: {self.log_file}")

            # Restore original stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr

            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


class TeeOutput:
    """Write to multiple output streams simultaneously."""

    def __init__(self, *streams):
        """Initialize with multiple output streams."""
        self.streams = streams

    def write(self, data):
        """Write data to all streams."""
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()  # Ensure immediate write
            except:
                pass  # Ignore errors on any stream

    def flush(self):
        """Flush all streams."""
        for stream in self.streams:
            try:
                stream.flush()
            except:
                pass


class GameStateLogger:
    """Tracks game state changes (fireworks, lives, clues) each turn."""

    def __init__(self, log_dir: str = "logs/game_state", game_id: Optional[int] = None):
        """Initialize game state logger.

        Args:
            log_dir: Directory to save game state logs
            game_id: Unique game identifier for parallel games
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include game_id in filename for parallel game tracking
        if game_id is not None:
            self.log_file = f"{log_dir}/game_state_{timestamp}_game{game_id}.jsonl"
        else:
            self.log_file = f"{log_dir}/game_state_{timestamp}.jsonl"

        self.episode_num = 0
        self.turn_num = 0

    def log_turn_state(self, observation: Dict[str, Any], turn_num: int, player_id: int, action: Optional[Dict] = None):
        """Log game state for a single turn.

        Args:
            observation: Player observation dict
            turn_num: Current turn number
            player_id: ID of the current player
            action: Action taken (optional)
        """
        state = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_num,
            "turn": turn_num,
            "current_player": player_id,
            "fireworks": observation.get('fireworks', {}),
            "fireworks_total": sum(observation.get('fireworks', {}).values()),
            "life_tokens": observation.get('life_tokens', 3),
            "information_tokens": observation.get('information_tokens', 8),
            "deck_size": observation.get('deck_size', 0),
        }

        if action:
            state["action"] = action

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(state) + '\n')
            f.flush()  # Ensure immediate write to disk

    def log_episode_start(self, episode_num: int):
        """Log the start of a new episode."""
        self.episode_num = episode_num
        self.turn_num = 0

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "EPISODE_START",
                "episode": episode_num
            }) + '\n')
            f.flush()  # Ensure immediate write to disk

    def log_episode_end(self, episode_num: int, final_score: int, max_fireworks: int, turns: int):
        """Log the end of an episode with final statistics."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "event": "EPISODE_END",
                "episode": episode_num,
                "final_score": final_score,
                "max_fireworks_reached": max_fireworks,
                "total_turns": turns
            }) + '\n')
            f.flush()  # Ensure immediate write to disk


class EventLogger:
    """Tracks specific events (life loss, clue gain/loss)."""

    def __init__(self, log_dir: str = "logs/events", game_id: Optional[int] = None):
        """Initialize event logger.

        Args:
            log_dir: Directory to save event logs
            game_id: Unique game identifier for parallel games
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include game_id in filename for parallel game tracking
        if game_id is not None:
            self.log_file = f"{log_dir}/events_{timestamp}_game{game_id}.jsonl"
        else:
            self.log_file = f"{log_dir}/events_{timestamp}.jsonl"

        # Track previous state to detect changes
        self.prev_lives = 3
        self.prev_clues = 8
        self.prev_fireworks_total = 0

        self.episode_num = 0
        self.turn_num = 0

    def check_and_log_changes(self, observation: Dict[str, Any], turn_num: int, player_id: int, action: Optional[Dict] = None):
        """Check for state changes and log events.

        Args:
            observation: Player observation dict
            turn_num: Current turn number
            player_id: ID of the current player
            action: Action taken (optional)
        """
        self.turn_num = turn_num

        current_lives = observation.get('life_tokens', 3)
        current_clues = observation.get('information_tokens', 8)
        current_fireworks = sum(observation.get('fireworks', {}).values())

        # Check for life loss
        if current_lives < self.prev_lives:
            self._log_event({
                "event_type": "LIFE_LOST",
                "turn": turn_num,
                "player": player_id,
                "lives_before": self.prev_lives,
                "lives_after": current_lives,
                "lives_lost": self.prev_lives - current_lives,
                "action": action
            })

        # Check for clue changes
        if current_clues < self.prev_clues:
            self._log_event({
                "event_type": "CLUE_USED",
                "turn": turn_num,
                "player": player_id,
                "clues_before": self.prev_clues,
                "clues_after": current_clues,
                "action": action
            })
        elif current_clues > self.prev_clues:
            self._log_event({
                "event_type": "CLUE_GAINED",
                "turn": turn_num,
                "player": player_id,
                "clues_before": self.prev_clues,
                "clues_after": current_clues,
                "reason": "played_5" if action and action.get('action_type') == 'PLAY' else "discard",
                "action": action
            })

        # Check for firework progress
        if current_fireworks > self.prev_fireworks_total:
            self._log_event({
                "event_type": "FIREWORK_ADVANCED",
                "turn": turn_num,
                "player": player_id,
                "fireworks_before": self.prev_fireworks_total,
                "fireworks_after": current_fireworks,
                "cards_played": current_fireworks - self.prev_fireworks_total,
                "current_stacks": observation.get('fireworks', {}),
                "action": action
            })

        # Update previous state
        self.prev_lives = current_lives
        self.prev_clues = current_clues
        self.prev_fireworks_total = current_fireworks

    def _log_event(self, event_data: Dict[str, Any]):
        """Log an event to file."""
        event_data["timestamp"] = datetime.now().isoformat()
        event_data["episode"] = self.episode_num

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_data) + '\n')
            f.flush()  # Ensure immediate write to disk

    def log_episode_start(self, episode_num: int):
        """Reset tracking for new episode."""
        self.episode_num = episode_num
        self.prev_lives = 3
        self.prev_clues = 8
        self.prev_fireworks_total = 0

        self._log_event({
            "event_type": "EPISODE_START",
            "episode": episode_num
        })

    def log_episode_end(self, episode_num: int, reason: str = "unknown"):
        """Log episode end."""
        self._log_event({
            "event_type": "EPISODE_END",
            "episode": episode_num,
            "end_reason": reason,
            "final_lives": self.prev_lives,
            "final_clues": self.prev_clues,
            "final_fireworks": self.prev_fireworks_total
        })
