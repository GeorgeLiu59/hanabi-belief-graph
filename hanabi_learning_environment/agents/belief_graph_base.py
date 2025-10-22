"""Base class for belief graph agents - Eliminates code duplication.

Implements Template Method pattern to provide shared infrastructure
while allowing variant-specific behavior through abstract methods.
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

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
        num_players = observation['num_players']
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
                self._process_single_event(event)
                self._seen_move_ids.add(move_id)

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
        """Robustly extract the JSON block returned by the LLM."""
        candidate = response_text

        if '```json' in response_text:
            start = response_text.find('```json') + len('```json')
            remainder = response_text[start:]
            end_marker = remainder.find('```')
            if end_marker != -1:
                candidate = remainder[:end_marker]
            else:
                candidate = remainder
        elif '```' in response_text:
            start = response_text.find('```') + 3
            remainder = response_text[start:]
            end_marker = remainder.find('```')
            if end_marker != -1:
                candidate = remainder[:end_marker]
            else:
                candidate = remainder

        candidate = candidate.strip()

        brace_count = 0
        json_start = -1
        for idx, char in enumerate(candidate):
            if char == '{':
                if brace_count == 0:
                    json_start = idx
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and json_start != -1:
                    snippet = candidate[json_start:idx + 1]
                    try:
                        json.loads(snippet)
                        return snippet
                    except json.JSONDecodeError:
                        continue

        raise ValueError("No valid JSON found in LLM response")

    @abstractmethod
    def _initialize_variant_specific_beliefs(self):
        """Initialize variant-specific belief structure."""
        pass

    @abstractmethod
    def _create_belief_entry(self) -> Dict[str, Any]:
        """Create a belief entry in the variant's format."""
        pass

    @abstractmethod
    def _process_single_event(self, event: Dict[str, Any]):
        """Process a single belief update event."""
        pass

    def _initialize_variant_specific_teammate_models(self, observation: Dict[str, Any]):
        """Initialize variant-specific teammate models - optional override."""
        pass

    def _generate_variant_specific_diff(self, before: Dict, after: Dict) -> str:
        """Generate variant-specific diff - optional override."""
        return ""