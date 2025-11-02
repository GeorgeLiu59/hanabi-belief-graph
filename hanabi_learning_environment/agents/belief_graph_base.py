"""Base class for belief graph agents - Eliminates code duplication.

Implements Template Method pattern to provide shared infrastructure
while allowing variant-specific behavior through abstract methods.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from ..agents.gemini_agent import GeminiAgent
from ..agents.agent_logger import AgentLogger
from ..agents.game_state_tracker import get_game_state_tracker
from ..agents.prompt_manager import PromptManager


class BeliefGraphAgentBase(GeminiAgent, ABC):
    """Base class for all belief graph agents.

    Provides shared infrastructure and template methods for
    variant-specific belief graph implementations.
    """

    def __init__(self, config, variant_name: str, agent_prefix: str, *args, **kwargs):
        """Initialize with common infrastructure."""
        super().__init__(config, *args, **kwargs)
        self.variant = variant_name
        self.agent_id = f"{agent_prefix}_{self.agent_id}"
        self.logger = AgentLogger(self.agent_id)
        self.prompt_manager = PromptManager()
        self.belief_graph = {}
        self._seen_move_ids: set[str] = set()
        self._previous_observed_hands = None

    def reset(self, config):
        """Reset and initialize belief graph."""
        super().reset(config)

        self.my_player_number = None

        self.belief_graph = {
            "GameState": {
                "clues": 8,
                "life": 3,
                "deck_size": 50
            },
            "My_Hand_Beliefs": {},
            "Teammate_Hand_Beliefs": {}
        }

        self._initialize_variant_specific_beliefs()

        self._seen_move_ids.clear()
        self._previous_observed_hands = None

        # Clear self-belief tracking variables
        if hasattr(self, '_previous_my_beliefs_snapshot'):
            delattr(self, '_previous_my_beliefs_snapshot')

        tracker = get_game_state_tracker()
        num_players = config.get('players', 2)
        tracker.reset(num_players)

    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with belief graph augmentation."""
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True

        self._update_game_state_tracker_early(observation)

        if observation.get('last_moves'):
            self._process_immediate_updates(observation)

        # CRITICAL: Update visible cards EVERY turn to sync "I see" data with reality
        self._update_visible_cards(observation)

        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None

        augmented_observation = observation.copy()
        tracker = get_game_state_tracker()
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = self.variant
        augmented_observation['belief_graph_natural_language'] = self.prompt_manager.format_belief_graph_natural_language(self.belief_graph, self.variant)

        self.logger.log_info("BELIEF_STATE", f"{self.variant.title()} graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))

        return super().act(augmented_observation)

    def _update_game_state_tracker_early(self, observation: Dict[str, Any]):
        """Update game state tracker with the latest observation IMMEDIATELY."""
        tracker = get_game_state_tracker()

        self.logger.log_info("EARLY_UPDATE", "🔄 Updating game state tracker with latest observation")

        tracker.update_visible_cards(
            self.my_player_number,
            observation.get('observed_hands', [])
        )

        if 'card_knowledge' in observation and len(observation['card_knowledge']) > 0:
            tracker.update_card_knowledge(
                self.my_player_number,
                observation['card_knowledge'][0]
            )

        tracker.update_game_state(
            observation.get('fireworks', {}),
            observation.get('life_tokens', 3),
            observation.get('information_tokens', 8),
            observation.get('deck_size', 50)
        )

        tracker_state = tracker.get_state_summary()
        self.logger.log_info("EARLY_UPDATE_COMPLETE", f"✅ Game state tracker updated: {tracker_state}")

        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size

        self.logger.log_info("BELIEF_SYNC", f"🔄 Belief graph synced: clues={tracker.clues}, lives={tracker.lives}, deck={tracker.deck_size}")

    def _determine_player_number(self, observation: Dict[str, Any]):
        """Determine which player we are based on the observation."""
        current_player = observation['current_player']
        current_player_offset = observation['current_player_offset']
        num_players = observation['num_players']

        self.my_player_number = (current_player - current_player_offset) % num_players
        self.logger.log_info("PLAYER_IDENTITY",
                           f"I am Player {self.my_player_number + 1} " +
                           f"(current_player={current_player}, offset={current_player_offset})")

    def _initialize_belief_graph_with_player_number(self, observation: Dict[str, Any]):
        """Initialize belief graph with correct player numbers."""
        my_player_id = self.my_player_number + 1
        num_players = observation['num_players']

        for i in range(5):
            self.belief_graph["My_Hand_Beliefs"][f"P{my_player_id}_Card{i+1}"] = self._create_belief_entry()

        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"] = {}
                for card in range(5):
                    self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"][f"P{player_id}_Card{card+1}"] = {
                        "actual_card_I_see": "Unknown",
                        f"p{player_id}_belief": self._create_belief_entry()
                    }

        self._initialize_variant_specific_teammate_models(observation)
        self._initialize_visible_cards(observation)

    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Initialize visible cards."""
        self._update_visible_cards(observation)

    def _update_visible_cards(self, observation: Dict[str, Any]):
        """Update visible cards from current observation - called every turn."""
        num_players = observation['num_players']

        # First, ensure belief graph structure matches current hand sizes
        self._sync_belief_graph_structure(observation)

        for offset, hand in enumerate(observation['observed_hands']):
            if offset == 0:
                continue

            actual_player_num = (self.my_player_number + offset) % num_players
            player_id = f"P{actual_player_num + 1}"

            if f"{player_id}_Hand" in self.belief_graph["Teammate_Hand_Beliefs"]:
                for card_idx, card in enumerate(hand):
                    if card.get('color') and card.get('rank') is not None:
                        card_key = f"{player_id}_Card{card_idx + 1}"
                        color = card['color']
                        rank = card['rank'] + 1 if card['rank'] >= 0 else '?'
                        self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"][card_key]["actual_card_I_see"] = f"{color} {rank}"

    def _sync_belief_graph_structure(self, observation: Dict[str, Any]):
        """Sync belief graph structure with current observation hand sizes."""
        num_players = observation['num_players']

        # Update my hand beliefs structure
        my_player_id = f"P{self.my_player_number + 1}"
        my_hand_size = len(observation['observed_hands'][0]) if observation['observed_hands'] else 0

        # Remove excess cards from my beliefs
        existing_my_cards = [k for k in self.belief_graph["My_Hand_Beliefs"].keys() if k.startswith(f"{my_player_id}_Card")]
        for card_key in existing_my_cards:
            card_num = int(card_key.split('Card')[1])
            if card_num > my_hand_size:
                del self.belief_graph["My_Hand_Beliefs"][card_key]

        # Add missing cards to my beliefs
        for i in range(my_hand_size):
            card_key = f"{my_player_id}_Card{i + 1}"
            if card_key not in self.belief_graph["My_Hand_Beliefs"]:
                self.belief_graph["My_Hand_Beliefs"][card_key] = self._create_belief_entry()
            else:
                # EXISTING CARD - Check if beliefs need reset due to card shuffle
                existing_belief = self.belief_graph["My_Hand_Beliefs"][card_key]

                # For my own cards, I can't see them directly, but I can detect shuffles
                # when the game state tracker reports hand changes and I have no recent clues
                # This is more complex, so for now just ensure structure is correct
                # The belief reset will happen through the normal belief update process
                pass

        # Update teammate hand structures with proper card position tracking
        for offset, hand in enumerate(observation['observed_hands']):
            if offset == 0:
                # Handle my own hand - SKIP if we already did immediate shifts
                my_player_id = f"P{self.my_player_number + 1}"

                # Check if we already processed immediate shifts for this round
                immediate_shift_already_done = False
                if hasattr(self, 'last_processed_moves') and self.last_processed_moves:
                    for move in self.last_processed_moves:
                        if move['move_type'] in ('PLAY', 'DISCARD') and move['player'] == self.my_player_number:
                            immediate_shift_already_done = True
                            break

                if immediate_shift_already_done:
                    self.logger.log_debug("SHIFT_SKIP_IMMEDIATE", f"Immediate shift already processed for {my_player_id}, skipping duplicate shift")
                else:
                    # Check if there are any PLAY/DISCARD events affecting my hand (legacy logic)
                    my_card_movement_detected = False
                    if hasattr(self, 'last_processed_moves') and self.last_processed_moves:
                        for move in self.last_processed_moves:
                            if move['move_type'] in ('PLAY', 'DISCARD') and move['player'] == self.my_player_number:
                                my_card_movement_detected = True
                                break

                    if my_card_movement_detected:
                        self.logger.log_debug("SHIFT_TRIGGER", f"Card movement detected for {my_player_id}, processing shifts")
                        self._handle_my_card_position_shifts(my_player_id, hand)
                    else:
                        self.logger.log_debug("SHIFT_SKIP", f"No card movement detected for {my_player_id}, skipping shifts")
                continue

            actual_player_num = (self.my_player_number + offset) % num_players
            player_id = f"P{actual_player_num + 1}"
            hand_size = len(hand)

            if f"{player_id}_Hand" not in self.belief_graph["Teammate_Hand_Beliefs"]:
                self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"] = {}

            # Remove excess cards
            existing_cards = [k for k in self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"].keys() if k.startswith(f"{player_id}_Card")]
            for card_key in existing_cards:
                card_num = int(card_key.split('Card')[1])
                if card_num > hand_size:
                    del self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"][card_key]

            # Track card movements and update beliefs accordingly
            self._handle_card_position_shifts(player_id, actual_player_num, hand)

        # Update variant-specific models that might need syncing
        self._update_variant_specific_models(observation)

    def _handle_card_position_shifts(self, player_id: str, actual_player_num: int, current_hand: List[Dict]):
        """Handle card position shifts by tracking card movements."""
        player_hand_key = f"{player_id}_Hand"

        # Build mapping of current cards to their positions
        current_card_positions = {}
        for i, card in enumerate(current_hand):
            if card.get('color') and card.get('rank') is not None:
                card_identity = f"{card['color']} {card['rank'] + 1}"
                current_card_positions[card_identity] = i + 1  # Card numbers are 1-based

        # Build mapping of old card positions from belief graph
        old_card_positions = {}
        for card_key, card_data in self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key].items():
            actual_card = card_data.get("actual_card_I_see", "")
            if actual_card and actual_card != "Unknown":
                card_num = int(card_key.split('Card')[1])
                old_card_positions[actual_card] = card_num

        # Find cards that moved positions
        moved_cards = {}
        new_cards = set(current_card_positions.keys()) - set(old_card_positions.keys())
        removed_cards = set(old_card_positions.keys()) - set(current_card_positions.keys())

        # Track moved cards
        for card_identity in current_card_positions:
            if card_identity in old_card_positions:
                old_pos = old_card_positions[card_identity]
                new_pos = current_card_positions[card_identity]
                if old_pos != new_pos:
                    moved_cards[card_identity] = (old_pos, new_pos)

        # Log movements for debugging
        if moved_cards:
            self.logger.log_info("CARD_POSITION_SHIFTS_DETECTED",
                f"Player {player_id} movements: {moved_cards}")

        if new_cards:
            self.logger.log_info("NEW_CARDS_DETECTED",
                f"Player {player_id} new cards: {new_cards}")

        if removed_cards:
            self.logger.log_info("REMOVED_CARDS_DETECTED",
                f"Player {player_id} removed cards: {removed_cards}")

        # Rebuild the belief graph with correct card positions
        new_hand_structure = {}

        # Process each position in the current hand
        for i, card in enumerate(current_hand):
            card_num = i + 1
            card_key = f"{player_id}_Card{card_num}"

            if card.get('color') and card.get('rank') is not None:
                card_identity = f"{card['color']} {card['rank'] + 1}"

                # Check if this is a moved card
                if card_identity in moved_cards:
                    old_pos, new_pos = moved_cards[card_identity]
                    old_card_key = f"{player_id}_Card{old_pos}"

                    # Move the existing belief data to the new position
                    if old_card_key in self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key]:
                        self.logger.log_info("MOVING_BELIEF_DATA",
                            f"Player {player_id}: {card_identity} moved from position {old_pos} to {new_pos}")

                        new_hand_structure[card_key] = self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key][old_card_key].copy()
                        new_hand_structure[card_key]["actual_card_I_see"] = card_identity
                    else:
                        # Fallback: create fresh belief
                        new_hand_structure[card_key] = {
                            "actual_card_I_see": card_identity,
                            f"p{actual_player_num}_belief": self._create_belief_entry()
                        }

                # Check if this is a completely new card
                elif card_identity in new_cards:
                    self.logger.log_info("CREATING_FRESH_BELIEF",
                        f"Player {player_id}: New card {card_identity} at position {card_num}")

                    new_hand_structure[card_key] = {
                        "actual_card_I_see": card_identity,
                        f"p{actual_player_num}_belief": self._create_belief_entry()  # FRESH BELIEFS!
                    }

                # Card stayed in place OR wasn't tracked properly - preserve existing beliefs
                else:
                    # Always try to preserve existing beliefs for this position
                    existing_key = f"{player_id}_Card{card_num}"
                    if existing_key in self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key]:
                        # Preserve existing belief data, just update the actual card
                        new_hand_structure[card_key] = self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key][existing_key].copy()
                        new_hand_structure[card_key]["actual_card_I_see"] = card_identity
                        self.logger.log_info("PRESERVING_EXISTING_BELIEFS",
                            f"Player {player_id}: Preserving beliefs for {card_identity} at position {card_num}")
                    else:
                        # No existing data found, create fresh
                        new_hand_structure[card_key] = {
                            "actual_card_I_see": card_identity,
                            f"p{actual_player_num}_belief": self._create_belief_entry()
                        }
                        self.logger.log_info("NO_EXISTING_BELIEFS_FOUND",
                            f"Player {player_id}: No existing beliefs for position {card_num}, creating fresh for {card_identity}")
            else:
                # Unknown card at this position
                card_key = f"{player_id}_Card{card_num}"
                # Try to preserve any existing belief data even if card is unknown
                existing_key = f"{player_id}_Card{card_num}"
                if existing_key in self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key]:
                    new_hand_structure[card_key] = self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key][existing_key].copy()
                    new_hand_structure[card_key]["actual_card_I_see"] = "Unknown"
                else:
                    new_hand_structure[card_key] = {
                        "actual_card_I_see": "Unknown",
                        f"p{actual_player_num}_belief": self._create_belief_entry()
                    }

        # Replace the entire hand structure with the corrected one
        self.belief_graph["Teammate_Hand_Beliefs"][player_hand_key] = new_hand_structure

    def _handle_my_card_position_shifts(self, my_player_id: str, current_hand: List[Dict]):
        """Handle shifting of self-beliefs when own cards are played/discarded and new cards are drawn.

        This method:
        1. Removes beliefs for played/discarded cards from specific positions
        2. Shifts remaining beliefs left to fill gaps
        3. Creates fresh beliefs for newly drawn cards
        4. Logs all changes with before/after states for full visibility
        """
        self.logger.log_debug("SHIFT_START", "Starting self card position shift processing")

        # Get current hand size
        current_hand_size = len(current_hand)
        self.logger.log_debug("SHIFT_CURRENT_HAND", f"Current hand size: {current_hand_size}")

        # Track if any cards were played/discarded from my hand
        my_card_removed = False
        my_old_hand_size = len(self.belief_graph.get("My_Hand_Beliefs", {}))

        # Log the before state
        before_beliefs = {}
        if "My_Hand_Beliefs" in self.belief_graph:
            for card_key, belief in self.belief_graph["My_Hand_Beliefs"].items():
                if card_key.startswith(f"{my_player_id}_Card"):
                    before_beliefs[card_key] = belief
        self.logger.log_debug("SHIFT_BEFORE_STATE", f"Before shift: {json.dumps(before_beliefs, indent=2)}")

        # Process plays/discards from MY hand to remove specific beliefs
        if hasattr(self, 'last_processed_moves') and self.last_processed_moves:
            for move in self.last_processed_moves:
                if move['move_type'] in ('PLAY', 'DISCARD') and move['player'] == self.my_player_number:
                    # Remove belief for the specific position that was played/discarded
                    target_pos = move['target_position']
                    self.logger.log_debug("SHIFT_CARD_REMOVAL", f"Removing {move['move_type']} belief from position {target_pos}")

                    if "My_Hand_Beliefs" in self.belief_graph:
                        # Remove the belief entry at the specific position
                        belief_key_to_remove = f"{my_player_id}_Card{target_pos + 1}"
                        if belief_key_to_remove in self.belief_graph["My_Hand_Beliefs"]:
                            removed_belief = self.belief_graph["My_Hand_Beliefs"][belief_key_to_remove]
                            del self.belief_graph["My_Hand_Beliefs"][belief_key_to_remove]
                            self.logger.log_info("SHIFT_CARD_REMOVED", f"Removed {move['move_type']} belief from position {target_pos}: {removed_belief}")
                            my_card_removed = True
                        else:
                            self.logger.log_warning("SHIFT_CARD_NOT_FOUND", f"No belief found at position {target_pos} to remove")

        # After removal, shift remaining beliefs left to fill gaps
        if my_card_removed and "My_Hand_Beliefs" in self.belief_graph:
            self.logger.log_debug("SHIFT_STARTING_LEFT_SHIFT", "Starting left shift of remaining beliefs")

            # Get all current belief positions and sort them
            current_positions = []
            for card_key in self.belief_graph["My_Hand_Beliefs"].keys():
                if card_key.startswith(f"{my_player_id}_Card"):
                    card_num = int(card_key.split('Card')[1])
                    current_positions.append(card_num)
            current_positions.sort()

            # Shift beliefs left to fill gaps by renumbering positions
            if current_positions:
                # Create new beliefs dict with shifted positions
                shifted_beliefs = {}
                for new_pos, old_pos in enumerate(current_positions, start=1):
                    old_key = f"{my_player_id}_Card{old_pos}"
                    new_key = f"{my_player_id}_Card{new_pos}"
                    if old_key in self.belief_graph["My_Hand_Beliefs"]:
                        shifted_beliefs[new_key] = self.belief_graph["My_Hand_Beliefs"][old_key]
                        if old_pos != new_pos:
                            self.logger.log_debug("SHIFT_SINGLE", f"Shifted belief from {old_key} to {new_key}: {shifted_beliefs[new_key]}")

                # Remove old position entries and add shifted ones
                keys_to_remove = [f"{my_player_id}_Card{pos}" for pos in current_positions]
                for key in keys_to_remove:
                    if key in self.belief_graph["My_Hand_Beliefs"]:
                        del self.belief_graph["My_Hand_Beliefs"][key]

                # Add shifted beliefs with new keys
                for new_key, belief in shifted_beliefs.items():
                    self.belief_graph["My_Hand_Beliefs"][new_key] = belief

                if len(current_positions) > 0:
                    old_positions_str = ", ".join([f"{my_player_id}_Card{pos}" for pos in current_positions])
                    new_positions_str = ", ".join(shifted_beliefs.keys())
                    self.logger.log_info("SHIFT_LEFT_SHIFT_COMPLETE", f"Completed left shift. {old_positions_str} → {new_positions_str}")

        # Create fresh beliefs for any new cards at the end
        if "My_Hand_Beliefs" in self.belief_graph:
            current_belief_count = len([k for k in self.belief_graph["My_Hand_Beliefs"].keys() if k.startswith(f"{my_player_id}_Card")])
            cards_to_add = current_hand_size - current_belief_count

            self.logger.log_debug("SHIFT_NEW_CARDS", f"Current belief count: {current_belief_count}, Hand size: {current_hand_size}, Cards to add: {cards_to_add}")

            for i in range(cards_to_add):
                new_pos = current_belief_count + i + 1
                new_belief = self._create_belief_entry()
                new_key = f"{my_player_id}_Card{new_pos}"
                self.belief_graph["My_Hand_Beliefs"][new_key] = new_belief
                self.logger.log_debug("SHIFT_FRESH_BELIEF", f"Created fresh belief for new card at {new_key}: {new_belief}")

        # Special case: if My_Hand_Beliefs doesn't exist or is empty, initialize it
        if "My_Hand_Beliefs" not in self.belief_graph or len(self.belief_graph["My_Hand_Beliefs"]) == 0:
            self.logger.log_info("SHIFT_INIT_EMPTY", "My_Hand_Beliefs is empty, initializing fresh beliefs")
            self.belief_graph["My_Hand_Beliefs"] = {}
            for i in range(current_hand_size):
                card_key = f"{my_player_id}_Card{i + 1}"
                self.belief_graph["My_Hand_Beliefs"][card_key] = self._create_belief_entry()

        # Log the after state
        after_beliefs = {}
        if "My_Hand_Beliefs" in self.belief_graph:
            for card_key, belief in self.belief_graph["My_Hand_Beliefs"].items():
                if card_key.startswith(f"{my_player_id}_Card"):
                    after_beliefs[card_key] = belief
        self.logger.log_debug("SHIFT_AFTER_STATE", f"After shift: {json.dumps(after_beliefs, indent=2)}")

        # Log summary of changes
        final_belief_count = len([k for k in self.belief_graph.get("My_Hand_Beliefs", {}).keys() if k.startswith(f"{my_player_id}_Card")])
        self.logger.log_info("SHIFT_SUMMARY", f"Shift summary: Hand size {my_old_hand_size} → {current_hand_size}, Cards removed: {my_card_removed}, Final belief count: {final_belief_count}")

        self.last_hand_size = current_hand_size

    def _summarize_belief(self, belief: Dict[str, Any]) -> str:
        """Create a concise summary of belief data for logging."""
        if 'possible_colors' in belief and 'possible_ranks' in belief:
            # Certainty-based belief
            colors = belief['possible_colors']
            ranks = belief['possible_ranks']
            return f"Certainty: {len(colors)}×{len(ranks)} possibilities (colors: {colors}, ranks: {ranks})"
        elif 'color_distribution' in belief and 'rank_distribution' in belief:
            # Probabilistic belief
            return f"Probabilistic: {len(belief['color_distribution'])} colors, {len(belief['rank_distribution'])} ranks"
        else:
            return f"Unknown belief type: {list(belief.keys())}"

    def _process_immediate_updates(self, observation: Dict[str, Any]):
        """Process IMMEDIATE belief updates after any game actions."""
        tracker = get_game_state_tracker()
        num_players = observation['num_players']

        last_moves = observation.get('last_moves', [])
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue

            move_data = last_move.get('move', {})
            action_type = move_data.get('action_type')

            if last_move.get('player', -1) < 0:
                continue

            # Process CLUE events (existing logic)
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                acting_player_offset = last_move['player']
                acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players

                target_offset = move_data['target_offset']
                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                if action_type == 'REVEAL_COLOR':
                    color = move_data['color']
                    card_indices = tracker.find_matching_cards_for_hint(
                        target_player_absolute, 'color', color
                    )

                    event = {
                        'clue_giver': f"P{acting_player_absolute + 1}",
                        'clue_recipient': f"P{target_player_absolute + 1}",
                        'clue_type': 'color',
                        'value': color,
                        'card_indices': card_indices
                    }

                elif action_type == 'REVEAL_RANK':
                    rank = move_data['rank']
                    card_indices = tracker.find_matching_cards_for_hint(
                        target_player_absolute, 'rank', rank
                    )

                    event = {
                        'clue_giver': f"P{acting_player_absolute + 1}",
                        'clue_recipient': f"P{target_player_absolute + 1}",
                        'clue_type': 'rank',
                        'value': rank,
                        'card_indices': card_indices
                    }

                self.logger.log_info("IMMEDIATE_UPDATE", f"Processing immediate {self.variant} belief update: {json.dumps(event, indent=2)}")
                self._process_single_event(event, observation)
                self._seen_move_ids.add(move_id)

            # Process PLAY/DISCARD events (NEW - update belief structure after card movements)
            elif action_type in ['PLAY', 'DISCARD']:
                acting_player_offset = last_move['player']
                acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players
                card_index = move_data.get('card_index', -1)
                player_id = f"P{acting_player_absolute + 1}"

                self.logger.log_info("CARD_MOVEMENT_UPDATE",
                    f"Player {player_id} {action_type.lower()} card at index {card_index}")

                # Track this move for later processing
                if not hasattr(self, 'last_processed_moves'):
                    self.last_processed_moves = []

                move_record = {
                    'move_type': action_type,
                    'player': acting_player_absolute,
                    'target_position': card_index
                }
                self.last_processed_moves.append(move_record)

                # Process belief shifts IMMEDIATELY for any player's card movement
                self._execute_immediate_belief_shift(player_id, acting_player_absolute, card_index, action_type)

                # If this was MY card played/discarded, also trigger belief structure update
                if acting_player_absolute == self.my_player_number:
                    self.logger.log_info("SELF_CARD_MOVEMENT",
                        f"My card movement detected - updating belief structure after {action_type.lower()}")

                self._seen_move_ids.add(move_id)

    def _execute_immediate_belief_shift(self, player_id: str, player_absolute: int, card_index: int, action_type: str):
        """Execute immediate belief shift when a card is played/discarded.

        For self: Handle immediate belief shifting
        For teammates: Let existing _handle_card_position_shifts handle it (no duplication)
        """
        self.logger.log_info("IMMEDIATE_SHIFT_START",
            f"Executing immediate belief shift for {player_id}: {action_type} card at index {card_index}")

        if player_absolute == self.my_player_number:
            # Handle self-belief shifting immediately
            self._execute_self_immediate_shift(player_id, card_index, action_type)
        else:
            # For teammates, let the existing _handle_card_position_shifts handle it
            # Don't duplicate - just log that we detected the event
            self.logger.log_debug("IMMEDIATE_SHIFT_TEAMMATE",
                f"Teammate {player_id} card movement detected - existing shift logic will handle")

        self.logger.log_info("IMMEDIATE_SHIFT_COMPLETE",
            f"Immediate belief shift complete for {player_id}")

    def _execute_self_immediate_shift(self, player_id: str, card_index: int, action_type: str):
        """Execute immediate belief shift for self's own cards."""
        self.logger.log_debug("IMMEDIATE_SELF_SHIFT",
            f"Processing immediate self shift: {action_type} card at position {card_index}")

        # Remove the specific belief that was played/discarded
        belief_key_to_remove = f"{player_id}_Card{card_index + 1}"

        if "My_Hand_Beliefs" in self.belief_graph and belief_key_to_remove in self.belief_graph["My_Hand_Beliefs"]:
            removed_belief = self.belief_graph["My_Hand_Beliefs"][belief_key_to_remove]
            del self.belief_graph["My_Hand_Beliefs"][belief_key_to_remove]
            self.logger.log_info("IMMEDIATE_SELF_REMOVED",
                f"Removed {action_type} belief: {belief_key_to_remove} was {self._summarize_belief(removed_belief)}")
        else:
            self.logger.log_warning("IMMEDIATE_SELF_NOT_FOUND",
                f"Belief not found for removal: {belief_key_to_remove}")

        # Get remaining beliefs and shift them left
        if "My_Hand_Beliefs" in self.belief_graph:
            current_positions = []
            for card_key in self.belief_graph["My_Hand_Beliefs"].keys():
                if card_key.startswith(f"{player_id}_Card"):
                    card_num = int(card_key.split('Card')[1])
                    current_positions.append(card_num)
            current_positions.sort()

            if current_positions:
                # Shift beliefs left to fill gaps
                shifted_beliefs = {}
                for new_pos, old_pos in enumerate(current_positions, start=1):
                    old_key = f"{player_id}_Card{old_pos}"
                    new_key = f"{player_id}_Card{new_pos}"
                    if old_key in self.belief_graph["My_Hand_Beliefs"]:
                        shifted_beliefs[new_key] = self.belief_graph["My_Hand_Beliefs"][old_key]
                        if old_pos != new_pos:
                            self.logger.log_debug("IMMEDIATE_SELF_SHIFT_SINGLE",
                                f"Shifted belief: {old_key} → {new_key}")

                # Remove old keys and add shifted ones
                keys_to_remove = [f"{player_id}_Card{pos}" for pos in current_positions]
                for key in keys_to_remove:
                    if key in self.belief_graph["My_Hand_Beliefs"]:
                        del self.belief_graph["My_Hand_Beliefs"][key]

                for new_key, belief in shifted_beliefs.items():
                    self.belief_graph["My_Hand_Beliefs"][new_key] = belief

                self.logger.log_info("IMMEDIATE_SELF_SHIFT_COMPLETE",
                    f"Self beliefs shifted: {len(current_positions)} beliefs repositioned")

        # Add a fresh belief for the new card at the end (hand size stays constant)
        if "My_Hand_Beliefs" in self.belief_graph:
            current_count = len([k for k in self.belief_graph["My_Hand_Beliefs"].keys() if k.startswith(f"{player_id}_Card")])
            new_belief = self._create_belief_entry()
            new_key = f"{player_id}_Card{current_count + 1}"
            self.belief_graph["My_Hand_Beliefs"][new_key] = new_belief
            self.logger.log_info("IMMEDIATE_SELF_FRESH_BELIEF",
                f"Created fresh belief: {new_key}")

    def _generate_belief_diff(self, before: Dict, after: Dict) -> str:
        """Generate a human-readable diff of belief changes."""
        diff_summary = "BELIEF CHANGES:\n"

        if 'My_Hand_Beliefs' in before and 'My_Hand_Beliefs' in after:
            for card_id in before['My_Hand_Beliefs']:
                if card_id in after['My_Hand_Beliefs']:
                    diff_summary += self._generate_card_belief_diff(
                        before['My_Hand_Beliefs'][card_id],
                        after['My_Hand_Beliefs'][card_id],
                        card_id
                    )

        if 'Teammate_Hand_Beliefs' in before and 'Teammate_Hand_Beliefs' in after:
            for player_hand in before['Teammate_Hand_Beliefs']:
                if player_hand in after['Teammate_Hand_Beliefs']:
                    before_hand = before['Teammate_Hand_Beliefs'][player_hand]
                    after_hand = after['Teammate_Hand_Beliefs'][player_hand]

                    for card_id in before_hand:
                        if card_id in after_hand:
                            for key in before_hand[card_id]:
                                if key.startswith('p') and key.endswith('_belief'):
                                    before_belief = before_hand[card_id][key]
                                    after_belief = after_hand[card_id][key]

                                    if before_belief != after_belief:
                                        diff_summary += f"  {player_hand}_{card_id}: Teammate belief updated\n"

        diff_summary += self._generate_variant_specific_diff(before, after)

        if diff_summary == "BELIEF CHANGES:\n":
            diff_summary += "  No significant changes detected\n"

        return diff_summary

    def _generate_card_belief_diff(self, before_card: Dict, after_card: Dict, card_id: str) -> str:
        """Generate diff for card beliefs - to be overridden by variants."""
        if 'possible_colors' in before_card:
            return self._generate_certainty_diff(before_card, after_card, card_id)
        elif 'color_distribution' in before_card:
            return self._generate_probabilistic_diff(before_card, after_card, card_id)
        return ""

    def _generate_certainty_diff(self, before_card: Dict, after_card: Dict, card_id: str) -> str:
        """Generate diff for certainty-based beliefs."""
        before_colors = set(before_card['possible_colors'])
        after_colors = set(after_card['possible_colors'])
        before_ranks = set(before_card['possible_ranks'])
        after_ranks = set(after_card['possible_ranks'])

        if before_colors != after_colors or before_ranks != after_ranks:
            removed_colors = before_colors - after_colors
            removed_ranks = before_ranks - after_ranks

            if removed_colors or removed_ranks:
                diff_summary = f"  {card_id}: "
                if removed_colors:
                    diff_summary += f"Eliminated colors: {list(removed_colors)} "
                if removed_ranks:
                    diff_summary += f"Eliminated ranks: {list(removed_ranks)} "
                diff_summary += f"({len(after_colors)}×{len(after_ranks)} possibilities remaining)\n"
                return diff_summary
        return ""

    def _generate_probabilistic_diff(self, before_card: Dict, after_card: Dict, card_id: str) -> str:
        """Generate diff for probabilistic beliefs."""
        before_color_dist = before_card['color_distribution']
        after_color_dist = after_card['color_distribution']
        before_rank_dist = before_card['rank_distribution']
        after_rank_dist = after_card['rank_distribution']

        changed = False
        changes = []

        for color in before_color_dist:
            if before_color_dist[color] != after_color_dist[color]:
                if after_color_dist[color] == 1.0:
                    changes.append(f"{color} → 100%")
                elif after_color_dist[color] == 0.0 and before_color_dist[color] > 0.0:
                    changes.append(f"{color} → 0%")
                changed = True

        for rank in before_rank_dist:
            if before_rank_dist[rank] != after_rank_dist[rank]:
                if after_rank_dist[rank] == 1.0:
                    changes.append(f"rank {rank} → 100%")
                elif after_rank_dist[rank] == 0.0 and before_rank_dist[rank] > 0.0:
                    changes.append(f"rank {rank} → 0%")
                changed = True

        if changed:
            return f"  {card_id}: {', '.join(changes)}\n"
        return ""

    def _extract_json_from_response(self, response_text: str) -> str:
        """Robustly extract the JSON block returned by the LLM with comprehensive fallbacks."""
        # First try: Extract from code blocks with JSON marker
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            if json_end == -1:
                json_end = len(response_text)
            json_text = response_text[json_start:json_end].strip()
            try:
                json.loads(json_text)
                return json_text
            except:
                pass  # Fall through to next method

        # Second try: Extract from any code blocks
        elif '```' in response_text:
            json_start = response_text.find('```') + 3
            json_end = response_text.find('```', json_start)
            if json_end == -1:
                json_end = len(response_text)
            json_text = response_text[json_start:json_end].strip()
            try:
                json.loads(json_text)
                return json_text
            except:
                pass  # Fall through to next method

        # Third try: Find JSON by brace counting (gv2 probabilistic approach)
        brace_count = 0
        json_start = -1
        json_end = -1

        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    json_start = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    json_end = i + 1
                    try:
                        potential_json = response_text[json_start:json_end]
                        json.loads(potential_json)
                        return potential_json
                    except:
                        continue

        # Fourth try: Find first { to last } (basic fallback for simple cases)
        if response_text.startswith('```'):
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                try:
                    json_text = response_text[json_start:json_end]
                    json.loads(json_text)
                    return json_text
                except:
                    pass

        raise ValueError("No valid JSON found in LLM response")

    @abstractmethod
    def _initialize_variant_specific_beliefs(self):
        """Initialize variant-specific belief structure."""
        pass

    def _update_variant_specific_models(self, observation: Dict[str, Any]):
        """Update variant-specific models that might need syncing - optional override."""
        pass

    @abstractmethod
    def _create_belief_entry(self) -> Dict[str, Any]:
        """Create a belief entry in the variant's format."""
        pass

    @abstractmethod
    def _process_single_event(self, event: Dict[str, Any], observation: Dict[str, Any]):
        """Process a single belief update event."""
        pass

    def _update_snapshots_after_belief_update(self):
        """Update all tracking snapshots after LLM belief updates to keep them in sync."""
        my_player_id = f"P{self.my_player_number + 1}"

        # Update self-belief snapshot
        if hasattr(self, '_previous_my_beliefs_snapshot'):
            self._previous_my_beliefs_snapshot = {}
            for card_key, belief_data in self.belief_graph["My_Hand_Beliefs"].items():
                if card_key.startswith(f"{my_player_id}_Card"):
                    self._previous_my_beliefs_snapshot[card_key] = belief_data.copy()

            self.logger.log_debug("SNAPSHOT_UPDATE",
                f"Updated self-belief snapshot with {len(self._previous_my_beliefs_snapshot)} cards")

        # Update teammate belief snapshots
        # This ensures that position shift tracking has the latest LLM-updated beliefs
        for player_hand_key, hand_data in self.belief_graph["Teammate_Hand_Beliefs"].items():
            player_id = player_hand_key.replace("_Hand", "")
            for card_key, card_data in hand_data.items():
                for belief_key, belief_data in card_data.items():
                    if belief_key.startswith('p') and belief_key.endswith('_belief'):
                        # Update the internal tracking that position shift logic uses
                        # This ensures that when cards move, the LLM-updated beliefs are preserved
                        self.logger.log_debug("TEAMMATE_SNAPSHOT_UPDATE",
                            f"Updated {player_id} {card_key} {belief_key} snapshot after LLM update")

        self.logger.log_info("SNAPSHOT_SYNC_COMPLETE",
            "All tracking snapshots updated after LLM belief update")

    def _initialize_variant_specific_teammate_models(self, observation: Dict[str, Any]):
        """Initialize variant-specific teammate models - optional override."""
        pass

    def _generate_variant_specific_diff(self, before: Dict, after: Dict) -> str:
        """Generate variant-specific diff - optional override."""
        return ""