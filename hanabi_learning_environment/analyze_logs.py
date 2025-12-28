#!/usr/bin/env python3
"""Helper script to navigate Hanabi game logs and locate events."""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


class LogNavigator:
    """Navigate Hanabi logs to locate specific events and game states."""

    def __init__(self, game_state_log: str = None, events_log: str = None):
        """Initialize with log file paths.

        Args:
            game_state_log: Path to game_state_*.jsonl file
            events_log: Path to events_*.jsonl file
        """
        self.game_state_log = game_state_log
        self.events_log = events_log

    def load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def locate_life_losses(self, events: List[Dict]):
        """Locate all life loss events in logs."""
        life_losses = [e for e in events if e.get('event_type') == 'LIFE_LOST']

        print(f"\n{'='*80}")
        print("LIFE LOSS LOCATIONS")
        print(f"{'='*80}")
        print(f"Found {len(life_losses)} life loss event(s)")

        if not life_losses:
            print("No lives lost.")
            return

        print("\nLocations:")
        for i, loss in enumerate(life_losses, 1):
            print(f"\n[{i}] Episode {loss['episode']}, Turn {loss['turn']}")
            print(f"    Player: {loss['player']}")
            print(f"    Lives: {loss['lives_before']} → {loss['lives_after']}")
            print(f"    Action: {json.dumps(loss.get('action', {}))}")
            print(f"    Log line: Episode {loss['episode']}, Turn {loss['turn']}")

    def locate_clue_events(self, events: List[Dict]):
        """Locate clue usage and gain events."""
        clue_used = [e for e in events if e.get('event_type') == 'CLUE_USED']
        clue_gained = [e for e in events if e.get('event_type') == 'CLUE_GAINED']

        print(f"\n{'='*80}")
        print("CLUE EVENT LOCATIONS")
        print(f"{'='*80}")
        print(f"Found {len(clue_used)} clue usage event(s)")
        print(f"Found {len(clue_gained)} clue gain event(s)")

        if clue_used:
            print("\nClue Usage Locations:")
            for i, use in enumerate(clue_used[:10], 1):  # Show first 10
                print(f"  [{i}] Episode {use['episode']}, Turn {use['turn']}")
                print(f"      Action: {json.dumps(use.get('action', {}))}")
            if len(clue_used) > 10:
                print(f"  ... and {len(clue_used) - 10} more")

        if clue_gained:
            print("\nClue Gain Locations:")
            for i, gain in enumerate(clue_gained[:10], 1):  # Show first 10
                print(f"  [{i}] Episode {gain['episode']}, Turn {gain['turn']}")
                print(f"      Reason: {gain.get('reason', 'unknown')}")
            if len(clue_gained) > 10:
                print(f"  ... and {len(clue_gained) - 10} more")

    def locate_episode_ends(self, events: List[Dict]):
        """Locate episode end events and their final states."""
        print(f"\n{'='*80}")
        print("EPISODE END LOCATIONS")
        print(f"{'='*80}")

        episode_ends = [e for e in events if e.get('event_type') == 'EPISODE_END']

        for end in episode_ends:
            print(f"\nEpisode {end['episode']}:")
            print(f"  End reason: {end.get('end_reason', 'unknown')}")
            print(f"  Final score: {end.get('final_fireworks', 0)}")
            print(f"  Final lives: {end.get('final_lives', 0)}")
            print(f"  Final clues: {end.get('final_clues', 0)}")

    def locate_game_state_at(self, states: List[Dict], episode: int, turn: int):
        """Locate game state at specific episode and turn."""
        print(f"\n{'='*80}")
        print(f"GAME STATE AT Episode {episode}, Turn {turn}")
        print(f"{'='*80}")

        matching = [s for s in states
                   if s.get('episode') == episode and s.get('turn') == turn]

        if not matching:
            print(f"No state found for Episode {episode}, Turn {turn}")
            return

        for state in matching:
            print(f"\nFireworks: {state.get('fireworks', {})}")
            print(f"Total score: {state.get('fireworks_total', 0)}")
            print(f"Lives: {state.get('life_tokens', 0)}")
            print(f"Clues: {state.get('information_tokens', 0)}")
            print(f"Deck: {state.get('deck_size', 0)}")
            print(f"Current player: {state.get('current_player', 0)}")
            if state.get('action'):
                print(f"Action: {json.dumps(state.get('action'), indent=2)}")

    def navigate_logs(self, episode: int = None, turn: int = None, event_type: str = None):
        """Navigate logs to find specific events or states."""
        if not self.events_log:
            print("Error: No events log provided")
            return

        events = self.load_jsonl(self.events_log)

        print("=" * 80)
        print("HANABI LOG NAVIGATOR")
        print("=" * 80)

        if event_type:
            print(f"\nSearching for event_type='{event_type}'")
            matching = [e for e in events if e.get('event_type') == event_type]
            print(f"Found {len(matching)} matching events")
            for i, event in enumerate(matching[:20], 1):
                print(f"\n[{i}] Episode {event.get('episode')}, Turn {event.get('turn', 'N/A')}")
                print(f"    {json.dumps(event, indent=4)}")
            if len(matching) > 20:
                print(f"\n... and {len(matching) - 20} more")
            return

        if episode is not None and turn is not None:
            if self.game_state_log:
                states = self.load_jsonl(self.game_state_log)
                self.locate_game_state_at(states, episode, turn)

            # Also show events at this location
            matching = [e for e in events
                       if e.get('episode') == episode and e.get('turn') == turn]
            if matching:
                print(f"\nEvents at this location:")
                for event in matching:
                    print(f"  {event.get('event_type')}: {json.dumps(event, indent=4)}")
            return

        # Default: show all event types available
        self.locate_life_losses(events)
        self.locate_clue_events(events)
        self.locate_episode_ends(events)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Navigate Hanabi game logs')
    parser.add_argument('--events', '-e', help='Path to events_*.jsonl file')
    parser.add_argument('--game-state', '-g', help='Path to game_state_*.jsonl file')
    parser.add_argument('--logs-dir', '-d', default='logs',
                       help='Directory containing logs (will auto-detect latest)')
    parser.add_argument('--episode', type=int, help='Episode number to examine')
    parser.add_argument('--turn', type=int, help='Turn number to examine (requires --episode)')
    parser.add_argument('--event-type', help='Event type to search for (e.g., LIFE_LOST, CLUE_USED)')

    args = parser.parse_args()

    # Auto-detect latest logs if not specified
    if not args.events and args.logs_dir:
        events_dir = Path(args.logs_dir) / 'events'
        if events_dir.exists():
            events_files = sorted(events_dir.glob('events_*.jsonl'), reverse=True)
            if events_files:
                args.events = str(events_files[0])
                print(f"Auto-detected events log: {args.events}")

    if not args.game_state and args.logs_dir:
        state_dir = Path(args.logs_dir) / 'game_state'
        if state_dir.exists():
            state_files = sorted(state_dir.glob('game_state_*.jsonl'), reverse=True)
            if state_files:
                args.game_state = str(state_files[0])
                print(f"Auto-detected game state log: {args.game_state}")

    navigator = LogNavigator(game_state_log=args.game_state, events_log=args.events)
    navigator.navigate_logs(episode=args.episode, turn=args.turn, event_type=args.event_type)


if __name__ == '__main__':
    main()
