"""Centralized Game State Tracker (Magic Orb)

This module provides a global source of truth for game state that all agents can update and query.
It resolves player identity confusion by maintaining absolute player IDs and visible card states.
"""

import json
import threading
from typing import Dict, List, Optional, Any
from collections import defaultdict


class GameStateTracker:
    """Global singleton that tracks game state across all agents."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.num_players = 0
        self.player_hands = {}
        self.player_card_knowledge = {}
        self.hint_history = []
        self.action_history = []
        self.move_id_set = set()
        self.fireworks = {}
        self.lives = 3
        self.clues = 8
        self.deck_size = 50
        self.resource_history = []
        self.state_update_count = 0

    def reset(self, num_players: int):
        """Reset tracker for new game."""
        self.num_players = num_players
        self.player_hands = {i: [] for i in range(num_players)}
        self.player_card_knowledge = {i: [] for i in range(num_players)}
        self.hint_history = []
        self.action_history = []
        self.move_id_set.clear()
        self.fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        self.lives = 3
        self.clues = 8
        self.deck_size = 50
        self.resource_history = []
        self.state_update_count = 0

    def update_visible_cards(self, observer_player_id: int, observed_hands: List[List[Dict]]):
        """Agent reports what cards they can see in other players' hands.

        Args:
            observer_player_id: Absolute player ID (0-indexed) of the observer
            observed_hands: List of hands visible to this observer (offset-indexed)
        """
        print(f"[GAME_STATE_TRACKER] 🔄 Updating visible cards for observer P{observer_player_id + 1}")
        print(f"[GAME_STATE_TRACKER] 👀 Observed hands: {len(observed_hands)} players")

        for offset, hand in enumerate(observed_hands):
            if offset == 0:
                continue

            target_player_id = (observer_player_id + offset) % self.num_players
            old_hand = self.player_hands.get(target_player_id, [])
            self.player_hands[target_player_id] = [
                {
                    'color': card.get('color'),
                    'rank': card.get('rank')
                }
                for card in hand
            ]

            # Log changes with detailed comparison
            if old_hand != self.player_hands[target_player_id]:
                print(f"[GAME_STATE_TRACKER] 🃏 Updated P{target_player_id + 1} hand:")
                print(f"  Before: {[(c.get('color', '?'), c.get('rank', '?')) for c in old_hand]}")
                print(f"  After:  {[(c.get('color', '?'), c.get('rank', '?')) for c in self.player_hands[target_player_id]]}")

                # Log each card change
                for idx, (old_card, new_card) in enumerate(zip(old_hand, self.player_hands[target_player_id])):
                    if old_card != new_card:
                        old_display = f"{old_card.get('color', '?')} {old_card.get('rank', '?')+1 if old_card.get('rank') is not None else '?'}"
                        new_display = f"{new_card.get('color', '?')} {new_card.get('rank', '?')+1 if new_card.get('rank') is not None else '?'}"
                        print(f"    Card {idx}: {old_display} → {new_display}")

    def update_card_knowledge(self, player_id: int, card_knowledge: List[Dict]):
        """Update what a player knows about their own cards (from card_knowledge).

        Args:
            player_id: Absolute player ID (0-indexed)
            card_knowledge: List of card knowledge dicts for this player's hand
        """
        self.player_card_knowledge[player_id] = [
            {
                'color': card.get('color'),
                'rank': card.get('rank')
            }
            for card in card_knowledge
        ]

    def update_game_state(
        self,
        fireworks: Dict,
        lives: int,
        clues: int,
        deck_size: int,
        turn_index: Optional[int] = None,
        last_moves: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """Update global game state.

        Args:
            fireworks: Dict mapping color to current level (e.g. {'R': 2, 'Y': 0, ...})
            lives: Remaining lives
            clues: Remaining clues
            deck_size: Cards left in deck
            turn_index: Optional turn/update counter for logging
            last_moves: Optional list of moves that led to this state

        Returns:
            List of resource change events detected during this update.
        """
        old_fireworks = self.fireworks.copy()
        old_lives = self.lives
        old_clues = self.clues
        old_deck_size = self.deck_size

        events: List[Dict[str, Any]] = []
        update_id = self.state_update_count + 1

        self.fireworks = fireworks.copy()
        self.lives = lives
        self.clues = clues
        self.deck_size = deck_size

        turn_label = turn_index if turn_index is not None else update_id
        last_move_summary = self._summarize_last_move(last_moves or [])

        # Log detailed changes
        changes = []
        if old_fireworks != self.fireworks:
            for color in ['R', 'Y', 'G', 'W', 'B']:
                if old_fireworks.get(color, 0) != self.fireworks.get(color, 0):
                    changes.append(f"Firework {color}: {old_fireworks.get(color, 0)} → {self.fireworks.get(color, 0)}")

        if old_lives != self.lives:
            changes.append(f"Lives: {old_lives} → {self.lives}")
            events.append({
                'turn': turn_label,
                'type': 'life',
                'before': old_lives,
                'after': self.lives,
                'change': self.lives - old_lives,
                'last_move': last_move_summary
            })

        if old_clues != self.clues:
            changes.append(f"Clues: {old_clues} → {self.clues}")
            events.append({
                'turn': turn_label,
                'type': 'clue',
                'before': old_clues,
                'after': self.clues,
                'change': self.clues - old_clues,
                'last_move': last_move_summary
            })

        if old_deck_size != self.deck_size:
            changes.append(f"Deck: {old_deck_size} → {self.deck_size}")

        if events:
            self.resource_history.extend(events)

        if changes:
            print(f"[GAME_STATE_TRACKER] 🎮 Game state updated:")
            for change in changes:
                print(f"  📊 {change}")

            # Summary state
            fireworks_total = sum(self.fireworks.values())
            print(f"[GAME_STATE_TRACKER] 📈 Current state: {fireworks_total} fireworks built, {self.lives} lives, {self.clues} clues, {self.deck_size} cards in deck")

        self.state_update_count = update_id
        return events

    def register_hint(
        self,
        giver_player_id: int,
        recipient_player_id: int,
        clue_type: str,
        value: Any,
        card_indices: List[int],
        move_signature: str
    ):
        """Register a hint action in the global history.

        Args:
            giver_player_id: Absolute player ID of hint giver (0-indexed)
            recipient_player_id: Absolute player ID of hint recipient (0-indexed)
            clue_type: 'color' or 'rank'
            value: The color or rank value
            card_indices: 0-indexed list of cards that match
            move_signature: Unique signature to prevent duplicates
        """
        if move_signature in self.move_id_set:
            return

        self.move_id_set.add(move_signature)

        hint_event = {
            'clue_giver': f"P{giver_player_id + 1}",
            'clue_recipient': f"P{recipient_player_id + 1}",
            'clue_type': clue_type,
            'value': value,
            'card_indices': card_indices
        }

        self.hint_history.append(hint_event)

        # Log hint registration with full context
        print(f"[GAME_STATE_TRACKER] 💡 Hint registered:")
        print(f"  Giver: P{giver_player_id + 1} → Recipient: P{recipient_player_id + 1}")
        print(f"  Type: {clue_type.upper()}, Value: {value}")
        print(f"  Card indices: {card_indices}")

        # Show recipient hand at time of hint
        recipient_hand = self.get_player_hand(recipient_player_id)
        hand_display = [(c.get('color', '?'), c.get('rank', '?')+1 if c.get('rank') is not None else '?') for c in recipient_hand]
        print(f"  Recipient hand: {hand_display}")

        # Show which cards actually match
        if recipient_hand and card_indices:
            print(f"  Matching cards:")
            for idx in card_indices:
                if idx < len(recipient_hand):
                    card = recipient_hand[idx]
                    color = card.get('color', '?')
                    rank = card.get('rank', '?')
                    if rank is not None:
                        rank += 1
                    print(f"    Card {idx}: {color} {rank}")

    def register_action(
        self,
        player_id: int,
        action_type: str,
        card_index: int,
        move_signature: str
    ):
        """Register a play/discard action.

        Args:
            player_id: Absolute player ID (0-indexed)
            action_type: 'PLAY' or 'DISCARD'
            card_index: Index of card played/discarded
            move_signature: Unique signature to prevent duplicates
        """
        if move_signature in self.move_id_set:
            return

        self.move_id_set.add(move_signature)

        action_event = {
            'type': action_type.lower(),
            'player': f"P{player_id + 1}",
            'card_index': card_index
        }

        self.action_history.append(action_event)

        # Log action registration with player hand context
        print(f"[GAME_STATE_TRACKER] 🎯 Action registered:")
        print(f"  Player: P{player_id + 1}, Action: {action_type.upper()}, Card index: {card_index}")

        # Show player hand before action if available
        player_hand = self.get_player_hand(player_id)
        if player_hand:
            hand_display = [(c.get('color', '?'), c.get('rank', '?')+1 if c.get('rank') is not None else '?') for c in player_hand]
            print(f"  Player hand: {hand_display}")

            if 0 <= card_index < len(player_hand):
                card = player_hand[card_index]
                color = card.get('color', '?')
                rank = card.get('rank', '?')
                if rank is not None:
                    rank += 1
                print(f"  Card acted upon: {color} {rank}")

    def get_recent_events(self, last_n: int = 10) -> List[Dict]:
        """Get recent hint events."""
        return self.hint_history[-last_n:]

    def get_player_hand(self, player_id: int) -> List[Dict]:
        """Get cards visible in a player's hand (as seen by others)."""
        return self.player_hands.get(player_id, [])

    def get_player_card_knowledge(self, player_id: int) -> List[Dict]:
        """Get what a player knows about their own cards."""
        return self.player_card_knowledge.get(player_id, [])

    def get_fireworks(self) -> Dict:
        """Get current fireworks state."""
        return self.fireworks.copy()

    def get_game_state(self) -> Dict:
        """Get complete game state."""
        return {
            'fireworks': self.fireworks.copy(),
            'lives': self.lives,
            'clues': self.clues,
            'deck_size': self.deck_size
        }

    def get_resource_history(self) -> List[Dict[str, Any]]:
        """Get history of life/clue token changes."""
        return list(self.resource_history)

    def _summarize_last_move(self, last_moves: List[Dict[str, Any]]) -> str:
        """Create a readable summary of the most recent move."""
        if not last_moves:
            return "No move context"

        most_recent = last_moves[0]
        player = most_recent.get('player')
        move = most_recent.get('move', {})
        action_type = move.get('action_type', 'UNKNOWN')
        player_display = f"P{player + 1}" if player is not None and player >= 0 else "P?"

        if action_type in ('REVEAL_COLOR', 'REVEAL_RANK'):
            target = move.get('target_offset')
            target_display = f"P{(player + target + 1) if player is not None and target is not None else '?'}"
            detail_key = 'color' if action_type == 'REVEAL_COLOR' else 'rank'
            detail_val = move.get(detail_key, '?')
            return f"{player_display} {action_type} {detail_val} → {target_display}"

        if action_type in ('PLAY', 'DISCARD'):
            card_idx = move.get('card_index', '?')
            color = most_recent.get('color', '?')
            rank = most_recent.get('rank', '?')
            outcome = "scored" if most_recent.get('scored') else "no score"
            return f"{player_display} {action_type} card {card_idx} ({color} {rank + 1 if isinstance(rank, int) else '?'}) {outcome}"

        return f"{player_display} {action_type}"

    def find_matching_cards_for_hint(
        self,
        recipient_player_id: int,
        clue_type: str,
        value: Any
    ) -> List[int]:
        """Find which cards in recipient's hand match the hint.

        Args:
            recipient_player_id: Absolute player ID (0-indexed)
            clue_type: 'color' or 'rank'
            value: Color string or rank value

        Returns:
            List of 0-indexed card positions that match
        """
        hand = self.get_player_hand(recipient_player_id)
        matching_indices = []

        for idx, card in enumerate(hand):
            if clue_type == 'color':
                card_color = card.get('color')
                if isinstance(card_color, str):
                    if card_color == value:
                        matching_indices.append(idx)
                elif isinstance(card_color, int) and card_color >= 0:
                    card_color_char = ['R', 'Y', 'G', 'W', 'B'][card_color]
                    if card_color_char == value:
                        matching_indices.append(idx)
            elif clue_type == 'rank':
                card_rank = card.get('rank')
                if card_rank == value:
                    matching_indices.append(idx)

        return matching_indices

    def get_state_summary(self) -> Dict:
        """Get current state summary for debugging."""
        return {
            'num_players': self.num_players,
            'player_hands': self.player_hands,
            'fireworks': self.fireworks,
            'lives': self.lives,
            'clues': self.clues,
            'deck_size': self.deck_size,
            'hint_count': len(self.hint_history),
            'action_count': len(self.action_history)
        }


_global_tracker = GameStateTracker()


def get_game_state_tracker() -> GameStateTracker:
    """Get the global game state tracker instance."""
    return _global_tracker
