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

    def update_visible_cards(self, observer_player_id: int, observed_hands: List[List[Dict]]):
        """Agent reports what cards they can see in other players' hands.

        Args:
            observer_player_id: Absolute player ID (0-indexed) of the observer
            observed_hands: List of hands visible to this observer (offset-indexed)
        """
        for offset, hand in enumerate(observed_hands):
            if offset == 0:
                continue

            target_player_id = (observer_player_id + offset) % self.num_players
            self.player_hands[target_player_id] = [
                {
                    'color': card.get('color'),
                    'rank': card.get('rank')
                }
                for card in hand
            ]

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

    def update_game_state(self, fireworks: Dict, lives: int, clues: int, deck_size: int):
        """Update global game state.

        Args:
            fireworks: Dict mapping color to current level (e.g. {'R': 2, 'Y': 0, ...})
            lives: Remaining lives
            clues: Remaining clues
            deck_size: Cards left in deck
        """
        self.fireworks = fireworks.copy()
        self.lives = lives
        self.clues = clues
        self.deck_size = deck_size

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
