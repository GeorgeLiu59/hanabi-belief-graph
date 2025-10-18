"""Belief graph scaffolding agents - EXACTLY as specified.

These are thin layers on GeminiAgent that maintain JSON belief structures
and pass them to the LLM for reasoning. No complex graph infrastructure needed.
"""

import json
from typing import Dict, Any, Optional, List

from ..agents.gemini_agent import GeminiAgent
from ..agents.agent_logger import AgentLogger
from ..agents.game_state_tracker import get_game_state_tracker
from ..agents.prompt_manager import PromptManager


class BeliefGraphCertaintyAgent(GeminiAgent):
    """Variation 1: The Certainty Graph
    
    Tracks possible values as sets (certainty-based).
    JSON Structure has My_Hand_Beliefs and Teammate_Hand_Beliefs.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.variant = 'certainty'
        self.agent_id = f"BG_certainty_{self.agent_id}"
        self.logger = AgentLogger(self.agent_id)
        self.prompt_manager = PromptManager()
        self.belief_graph = {}
        # 使用已处理 move 的唯一哈希集合进行去重，避免因历史长度相同而漏检
        self._seen_move_ids: set[str] = set()
        # 追踪上一帧的 observed_hands 来检测新牌
        self._previous_observed_hands = None
    
    def reset(self, config):
        """Reset and initialize belief graph."""
        super().reset(config)

        # We need to determine our player number first
        self.my_player_number = None  # Will be set on first observation

        # Initialize belief graph with just the game state
        # My_Hand_Beliefs and Teammate_Hand_Beliefs will be initialized
        # once we know which player we are
        self.belief_graph = {
            "GameState": {
                "clues": 8,
                "life": 3,
                "deck_size": 50
            },
            "My_Hand_Beliefs": {},
            "Teammate_Hand_Beliefs": {}
        }

        # 清空已处理 move 集合
        self._seen_move_ids.clear()
        # 清空上一帧 observed_hands
        self._previous_observed_hands = None

        # Initialize game state tracker
        tracker = get_game_state_tracker()
        num_players = config.get('players', 2)
        tracker.reset(num_players)
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with belief graph augmentation."""
        # Determine our player number on first observation
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True

        # CRITICAL: Update game state tracker IMMEDIATELY at the start of every act call
        # This ensures the tracker has the latest possible information before any processing
        self._update_game_state_tracker_early(observation)

        # Process belief updates IMMEDIATELY after any hints
        if observation.get('last_moves'):
            self._process_immediate_updates(observation)

        # Always keep teammate actual cards up to date
        self._refresh_visible_cards(observation)

        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None

        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
        # Update GameState with current fireworks from tracker
        tracker = get_game_state_tracker()
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'certainty'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()

        # Log current belief state
        self.logger.log_info("BELIEF_STATE", f"Certainty graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))

        # Let parent handle all the action logic
        return super().act(augmented_observation)
    
    # ------------------------------------------------------------------
    # New helper: Update game state tracker at the earliest possible point
    # ------------------------------------------------------------------
    def _update_game_state_tracker_early(self, observation: Dict[str, Any]):
        """Update game state tracker with the latest observation IMMEDIATELY.

        This is called at the very start of every act() call to ensure the tracker
        has the most current game state before any belief updates or processing.
        """
        tracker = get_game_state_tracker()

        self.logger.log_info("EARLY_UPDATE", "🔄 Updating game state tracker with latest observation")

        # Update visible cards immediately
        tracker.update_visible_cards(
            self.my_player_number,
            observation.get('observed_hands', [])
        )

        # Update card knowledge immediately
        if 'card_knowledge' in observation and len(observation['card_knowledge']) > 0:
            tracker.update_card_knowledge(
                self.my_player_number,
                observation['card_knowledge'][0]
            )

        # Update game state immediately
        tracker.update_game_state(
            observation.get('fireworks', {}),
            observation.get('life_tokens', 3),
            observation.get('information_tokens', 8),
            observation.get('deck_size', 50)
        )

        # Log the updated tracker state
        tracker_state = tracker.get_state_summary()
        self.logger.log_info("EARLY_UPDATE_COMPLETE",
                           f"✅ Game state tracker updated: {tracker_state}")

        # Update belief graph immediately with fresh tracker data
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size

        self.logger.log_info("BELIEF_SYNC", f"🔄 Belief graph synced: clues={tracker.clues}, lives={tracker.lives}, deck={tracker.deck_size}")

    # ------------------------------------------------------------------
    # New helper: refresh teammate actual cards each observation to avoid
    # stale duplicates when hands change after plays/discards
    # ------------------------------------------------------------------
    def _refresh_visible_cards(self, observation):
        """Update actual_card_I_see fields for teammate hands using tracker."""
        tracker = get_game_state_tracker()
        num_players = tracker.num_players
        self.logger.log_debug("CARD_REFRESH", "Starting card refresh check...")

        # CRITICAL: Update tracker with current observation BEFORE reading from it
        tracker.update_visible_cards(
            self.my_player_number,
            observation.get('observed_hands', [])
        )

        current_hands = observation.get('observed_hands', [])
        
        # Check for PLAY/DISCARD actions that cause hand shifts
        # When I play/discard, all cards to the right shift left, so we need to reset beliefs
        if 'last_moves' in observation:
            for move_entry in observation['last_moves']:
                player_offset = move_entry.get('player', -1)
                if player_offset < 0:
                    continue

                # Calculate absolute player number
                acting_player_abs = (self.my_player_number + player_offset) % num_players
                
                move = move_entry.get('move', {})
                action_type = move.get('action_type')
                
                # If I played or discarded a card, SHIFT beliefs left (like the game does)
                if action_type in ['PLAY', 'DISCARD'] and acting_player_abs == self.my_player_number:
                    card_idx = move.get('card_index', -1)
                    if card_idx >= 0:
                        my_player_id = self.my_player_number + 1
                        self.logger.log_info("BELIEF_SHIFT", 
                            f"🔀 Shifting MY hand beliefs left after {action_type} at index {card_idx}")
                        
                        # Save beliefs from cards that will shift
                        beliefs_to_shift = []
                        for idx in range(card_idx + 1, 5):
                            old_key = f"P{my_player_id}_Card{idx + 1}"
                            if old_key in self.belief_graph.get("My_Hand_Beliefs", {}):
                                beliefs_to_shift.append(self.belief_graph["My_Hand_Beliefs"][old_key].copy())
                        
                        # Shift beliefs left: Card(i+1) → Card(i)
                        for i, belief in enumerate(beliefs_to_shift):
                            new_key = f"P{my_player_id}_Card{card_idx + i + 1}"
                            if new_key in self.belief_graph["My_Hand_Beliefs"]:
                                self.belief_graph["My_Hand_Beliefs"][new_key] = belief
                                self.logger.log_debug("BELIEF_SHIFT", 
                                    f"  Moved belief from Card{card_idx + i + 2} to Card{card_idx + i + 1}")
                        
                        # Reset ONLY the last card (new card drawn from deck)
                        last_card_key = f"P{my_player_id}_Card5"
                        if last_card_key in self.belief_graph["My_Hand_Beliefs"]:
                            old_belief = self.belief_graph["My_Hand_Beliefs"][last_card_key].copy()
                            self._reset_card_belief(
                                self.belief_graph["My_Hand_Beliefs"][last_card_key],
                                last_card_key,
                                f"New card drawn after {action_type}"
                            )
                            self.logger.log_info("BELIEF_RESET_DETAIL", 
                                f"  Before: colors={old_belief.get('possible_colors')}, ranks={old_belief.get('possible_ranks')}\n" +
                                f"  After:  colors={self.belief_graph['My_Hand_Beliefs'][last_card_key].get('possible_colors')}, " +
                                f"ranks={self.belief_graph['My_Hand_Beliefs'][last_card_key].get('possible_ranks')}")
        
        # Handle TEAMMATE hands using tracker data
        for player_num in range(num_players):
            if player_num == self.my_player_number:
                continue

            player_id = f"P{player_num + 1}"
            hand_key = f"{player_id}_Hand"

            if hand_key not in self.belief_graph.get("Teammate_Hand_Beliefs", {}):
                continue

            tracker_hand = tracker.get_player_hand(player_num)

            for card_idx, card in enumerate(tracker_hand):
                card_key = f"{player_id}_Card{card_idx + 1}"

                if card_key not in self.belief_graph["Teammate_Hand_Beliefs"][hand_key]:
                    continue

                color = card.get('color')
                rank = card.get('rank')

                if color is not None and rank is not None:
                    rank_display = rank + 1 if rank >= 0 else '?'
                    new_actual = f"{color} {rank_display}"
                    self.belief_graph["Teammate_Hand_Beliefs"][hand_key][card_key]["actual_card_I_see"] = new_actual
                else:
                    self.belief_graph["Teammate_Hand_Beliefs"][hand_key][card_key]["actual_card_I_see"] = "Unknown"
        
        # Also check for teammate PLAY/DISCARD actions and reset their beliefs accordingly
        if 'last_moves' in observation:
            for move_entry in observation['last_moves']:
                player_offset = move_entry.get('player', -1)
                if player_offset < 0:
                    continue
                
                acting_player_abs = (self.my_player_number + player_offset) % num_players
                move = move_entry.get('move', {})
                action_type = move.get('action_type')
                
                # If teammate played/discarded, SHIFT their beliefs left (like the game does)
                if action_type in ['PLAY', 'DISCARD'] and acting_player_abs != self.my_player_number:
                    card_idx = move.get('card_index', -1)
                    if card_idx >= 0:
                        player_id = f"P{acting_player_abs + 1}"
                        hand_key = f"{player_id}_Hand"
                        
                        if hand_key in self.belief_graph.get("Teammate_Hand_Beliefs", {}):
                            self.logger.log_info("BELIEF_SHIFT", 
                                f"🔀 Shifting TEAMMATE {player_id} beliefs left after {action_type} at index {card_idx}")
                            
                            belief_field = f"p{acting_player_abs + 1}_belief"
                            
                            # Save beliefs from cards that will shift
                            beliefs_to_shift = []
                            for idx in range(card_idx + 1, 5):
                                old_card_key = f"{player_id}_Card{idx + 1}"
                                if old_card_key in self.belief_graph["Teammate_Hand_Beliefs"][hand_key]:
                                    if belief_field in self.belief_graph["Teammate_Hand_Beliefs"][hand_key][old_card_key]:
                                        beliefs_to_shift.append(
                                            self.belief_graph["Teammate_Hand_Beliefs"][hand_key][old_card_key][belief_field].copy()
                                        )
                            
                            # Shift beliefs left: Card(i+1) → Card(i)
                            for i, belief in enumerate(beliefs_to_shift):
                                new_card_key = f"{player_id}_Card{card_idx + i + 1}"
                                if new_card_key in self.belief_graph["Teammate_Hand_Beliefs"][hand_key]:
                                    if belief_field in self.belief_graph["Teammate_Hand_Beliefs"][hand_key][new_card_key]:
                                        self.belief_graph["Teammate_Hand_Beliefs"][hand_key][new_card_key][belief_field] = belief
                                        self.logger.log_debug("BELIEF_SHIFT", 
                                            f"  Moved {player_id} belief from Card{card_idx + i + 2} to Card{card_idx + i + 1}")
                            
                            # Reset ONLY the last card (new card drawn from deck)
                            last_card_key = f"{player_id}_Card5"
                            if last_card_key in self.belief_graph["Teammate_Hand_Beliefs"][hand_key]:
                                if belief_field in self.belief_graph["Teammate_Hand_Beliefs"][hand_key][last_card_key]:
                                    old_belief = self.belief_graph["Teammate_Hand_Beliefs"][hand_key][last_card_key][belief_field].copy()
                                    self._reset_card_belief(
                                        self.belief_graph["Teammate_Hand_Beliefs"][hand_key][last_card_key][belief_field],
                                        last_card_key,
                                        f"Teammate's new card drawn after {action_type}"
                                    )
                                    new_belief = self.belief_graph["Teammate_Hand_Beliefs"][hand_key][last_card_key][belief_field]
                                    self.logger.log_info("BELIEF_RESET_DETAIL", 
                                        f"  Before: colors={old_belief.get('possible_colors')}, ranks={old_belief.get('possible_ranks')}\n" +
                                        f"  After:  colors={new_belief.get('possible_colors')}, ranks={new_belief.get('possible_ranks')}")
        
        # Store current hands for next comparison
        self._previous_observed_hands = [hand.copy() if isinstance(hand, list) else hand for hand in current_hands] if current_hands else None

    def _process_immediate_updates(self, observation: Dict[str, Any]):
        """Process IMMEDIATE belief updates after any game actions."""
        tracker = get_game_state_tracker()
        num_players = observation['num_players']

        # The tracker is ALREADY updated in _update_game_state_tracker_early()
        # No need to update it again here - use the current state

        # NOW process hint events with updated tracker data
        last_moves = observation.get('last_moves', [])
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue

            move_data = last_move.get('move', {})
            action_type = move_data.get('action_type')

            if last_move.get('player', -1) < 0:
                continue

            # Process hint events immediately
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                acting_player_offset = last_move['player']
                acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players

                target_offset = move_data['target_offset']
                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                if action_type == 'REVEAL_COLOR':
                    color = move_data['color']
                    # Tracker has correct data to find matching cards
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
                    # Tracker has correct data to find matching cards
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

                # Process the hint event IMMEDIATELY
                self.logger.log_info("IMMEDIATE_UPDATE", f"Processing immediate belief update: {json.dumps(event, indent=2)}")
                self._process_single_event(event)
                self._seen_move_ids.add(move_id)

    def _reset_card_belief(self, belief_entry: Dict, card_key: str = "Unknown", reason: str = ""):
        """Reset a certainty belief entry back to full uncertainty (5 colors × 5 ranks)."""
        self.logger.log_info("BELIEF_RESET", f"🔄 Resetting {card_key} belief to full uncertainty. Reason: {reason}")
        belief_entry["possible_colors"] = ["red", "blue", "green", "white", "yellow"]
        belief_entry["possible_ranks"] = [1, 2, 3, 4, 5]
    
    def _format_belief_graph_natural_language(self) -> str:
        """Convert belief graph to natural language for easier LLM understanding."""
        nl_description = "## BELIEF GRAPH ANALYSIS (CERTAINTY VARIANT)\n\n"
        
        # Game state
        gs = self.belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"
        
        # My hand beliefs
        nl_description += "**MY HAND BELIEFS:**\n"
        nl_description += "⚠️ CRITICAL: Use 'card_index' in your action! Card 1 below = card_index 0, Card 2 = card_index 1, etc.\n\n"
        
        # Get current fireworks state to verify playability
        game_state = self.belief_graph.get('GameState', {})
        fireworks = game_state.get('fireworks', {})
        if not fireworks:
            # Fallback: try to get from observation if available
            fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        
        for card_id, beliefs in self.belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            card_idx = int(card_num) - 1  # Convert to 0-indexed
            colors = beliefs['possible_colors']
            ranks = beliefs['possible_ranks']
            
            if len(colors) == 1 and len(ranks) == 1:
                # 100% certain - check if actually playable
                color = colors[0]
                rank = ranks[0]
                color_key = color[0].upper() if len(color) > 0 else color.upper()
                current_firework = fireworks.get(color_key, 0)
                
                if rank == current_firework + 1:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): ✅ PLAYABLE NOW - {color.upper()} {rank} matches next needed ({color_key} firework at {current_firework}) → USE card_index {card_idx}!\n"
                elif rank <= current_firework:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): ❌ ALREADY PLAYED - {color.upper()} {rank} (firework already at {current_firework})\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): ⏳ NOT YET - {color.upper()} {rank} (need {current_firework + 1} first)\n"
            elif len(ranks) == 1 and ranks[0] == 1:
                # Rank 1 with ANY color uncertainty - check if any unstarted color
                possible_plays = []
                for c in colors:
                    c_key = c[0].upper() if len(c) > 0 else c.upper()
                    if fireworks.get(c_key, 0) == 0:
                        possible_plays.append(c.upper())
                
                if possible_plays:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): ⚠️ PLAYABLE NOW - Rank 1, can start {'/'.join(possible_plays)} (unstarted colors)\n"
                    nl_description += f"  → PLAY using card_index {card_idx} - 1s always work at game start!\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): ❌ ALREADY STARTED - Rank 1 but all possible colors {'/'.join(colors).upper()} already started\n"
            elif len(colors) == 1 and len(ranks) <= 2:
                # Known color with 1-2 rank possibilities - can be strategic
                color = colors[0]
                nl_description += f"- Card {card_num} (card_index {card_idx}): ⚠️ MAYBE PLAYABLE - {color.upper()} {'/'.join(map(str,ranks))}\n"
                nl_description += f"  → Check fireworks; if playable, use card_index {card_idx}\n"
            elif len(colors) == 1:
                nl_description += f"- Card {card_num} (card_index {card_idx}): ❌ RISKY - only know color {colors[0].upper()}\n"
            elif len(ranks) == 1:
                nl_description += f"- Card {card_num} (card_index {card_idx}): ❌ RISKY - only know rank {ranks[0]}\n"
            else:
                nl_description += f"- Card {card_num} (card_index {card_idx}): ❌ TOO RISKY - too uncertain\n"
        
        # Teammate beliefs
        nl_description += "\n**🎯 ACTION GUIDANCE:**\n"
        nl_description += "✅ CERTAIN cards: Play immediately with confidence!\n"
        nl_description += "⚠️ SAFE BET cards: Calculated risks worth taking (especially rank 1s)\n"
        nl_description += "⚠️ MAYBE PLAYABLE: Verify against fireworks before playing\n"
        nl_description += "❌ RISKY/TOO RISKY: Don't play - give hints or discard instead\n\n"
        
        nl_description += "**TEAMMATE KNOWLEDGE MODEL:**\n"
        for player_hand, hand_data in self.belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            # Show from MY perspective (Player 1 sees Player 2's cards)
            nl_description += f"\nPlayer {player_num}'s cards (what I see vs what they know):\n"
            
            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')
                belief = card_data.get(f'p{player_num}_belief', {})
                colors = belief.get('possible_colors', [])
                ranks = belief.get('possible_ranks', [])
                
                if len(colors) == 1 and len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they KNOW it's {colors[0].upper()} {ranks[0]}\n"
                elif len(colors) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know COLOR {colors[0].upper()} only\n"
                elif len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know RANK {ranks[0]} only\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know NOTHING\n"
        
        return nl_description
    
    def _determine_player_number(self, observation: Dict[str, Any]):
        """Determine which player we are based on the observation.
        
        The key insight: current_player_offset tells us how many seats away 
        the current player is from us (the observer).
        
        So: my_player_number = (current_player - current_player_offset) % num_players
        """
        current_player = observation['current_player']
        current_player_offset = observation['current_player_offset']
        num_players = observation['num_players']
        
        # Calculate our player number
        self.my_player_number = (current_player - current_player_offset) % num_players
        self.logger.log_info("PLAYER_IDENTITY", 
                           f"I am Player {self.my_player_number + 1} " +
                           f"(current_player={current_player}, offset={current_player_offset})")
    
    def _initialize_belief_graph_with_player_number(self, observation: Dict[str, Any]):
        """Initialize belief graph with correct player numbers."""
        my_player_id = self.my_player_number + 1  # Convert to 1-indexed
        num_players = observation['num_players']
        
        # Initialize my hand beliefs
        for i in range(5):  # Assuming 5 cards per hand
            self.belief_graph["My_Hand_Beliefs"][f"P{my_player_id}_Card{i+1}"] = {
                "possible_colors": ["red", "blue", "green", "white", "yellow"],
                "possible_ranks": [1, 2, 3, 4, 5]
            }
        
        # Initialize teammate beliefs
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"] = {}
                for card in range(5):
                    self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"][f"P{player_id}_Card{card+1}"] = {
                        "actual_card_I_see": "Unknown",
                        f"p{player_id}_belief": {
                            "possible_colors": ["red", "blue", "green", "white", "yellow"],
                            "possible_ranks": [1, 2, 3, 4, 5]
                        }
                    }
        
        # Initialize visible cards
        self._initialize_visible_cards(observation)
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Update belief graph with visible cards from tracker."""
        tracker = get_game_state_tracker()
        num_players = tracker.num_players

        for player_num in range(num_players):
            if player_num == self.my_player_number:
                continue

            player_id = f"P{player_num + 1}"
            hand_key = f"{player_id}_Hand"

            if hand_key not in self.belief_graph["Teammate_Hand_Beliefs"]:
                continue

            tracker_hand = tracker.get_player_hand(player_num)

            for card_idx, card in enumerate(tracker_hand):
                card_key = f"{player_id}_Card{card_idx + 1}"
                color = card.get('color')
                rank = card.get('rank')

                if color and rank is not None:
                    rank_display = rank + 1 if rank >= 0 else '?'
                    self.belief_graph["Teammate_Hand_Beliefs"][hand_key][card_key]["actual_card_I_see"] = f"{color} {rank_display}"
    
    def _update_beliefs_via_llm(self, observation: Dict[str, Any]):
        """Update beliefs using LLM as specified."""
        # (1) Log observation going in (skip non-serializable pyhanabi object and verbose vectorized field)
        obs_for_logging = {k: v for k, v in observation.items() if k not in ['pyhanabi', 'vectorized']}
        self.logger.log_debug("OBSERVATION_IN", f"Observation: {json.dumps(obs_for_logging, indent=2)}")
        
        # Detect all events that happened
        events = self._detect_events(observation)
        if not events:
            return
        
        # Process each event one by one
        for i, event in enumerate(events):
            # (2) Log event coming out
            self.logger.log_info("EVENT_OUT", f"Detected event {i+1}/{len(events)}: {json.dumps(event, indent=2)}")
            self._process_single_event(event)
    
    def _process_single_event(self, event: Dict[str, Any]):
        """Process a single event through the LLM."""
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        # (3) Log belief graph before LLM update
        self.logger.log_info("BELIEF_BEFORE_UPDATE", f"Current belief graph: {json.dumps(self.belief_graph, indent=2)}")

        # Use intelligent game rules prompt
        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'certainty')

        # (4) Log prompt into LLM
        self.logger.log_info("PROMPT_TO_LLM", f"Prompt ({len(prompt)} chars):\n{prompt}")

        try:
            # Get the full reasoning response
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2}
            )

            # (5) Log LLM's output
            self.logger.log_info("LLM_OUTPUT", f"LLM Response:\n{reasoning_response.text}")

            response_text = reasoning_response.text
            json_text = self._extract_json_from_response(response_text)

            self.logger.log_debug("BELIEF_UPDATE_JSON_EXTRACTED", json_text)

            # Update with new belief state
            new_belief_graph = json.loads(json_text)

            # Generate and log diff BEFORE updating
            diff_summary = self._generate_belief_diff(self.belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            # Now update the belief graph
            self.belief_graph = new_belief_graph

            # (6) Log belief graph after update
            self.logger.log_info("BELIEF_AFTER_UPDATE", f"Updated belief graph:\n{json.dumps(self.belief_graph, indent=2)}")

        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update beliefs: {e}")
    
    def _generate_belief_diff(self, before: Dict, after: Dict) -> str:
        """Generate a human-readable diff of belief changes."""
        diff_summary = "BELIEF CHANGES:\n"
        
        # Check My_Hand_Beliefs for changes
        if 'My_Hand_Beliefs' in before and 'My_Hand_Beliefs' in after:
            for card_id in before['My_Hand_Beliefs']:
                if card_id in after['My_Hand_Beliefs']:
                    before_card = before['My_Hand_Beliefs'][card_id]
                    after_card = after['My_Hand_Beliefs'][card_id]
                    
                    # For certainty variant - check possible values
                    if 'possible_colors' in before_card:
                        before_colors = set(before_card['possible_colors'])
                        after_colors = set(after_card['possible_colors'])
                        before_ranks = set(before_card['possible_ranks'])
                        after_ranks = set(after_card['possible_ranks'])
                        
                        if before_colors != after_colors or before_ranks != after_ranks:
                            removed_colors = before_colors - after_colors
                            removed_ranks = before_ranks - after_ranks

                            if removed_colors or removed_ranks:
                                diff_summary += f"  {card_id}: "
                                if removed_colors:
                                    diff_summary += f"Eliminated colors: {list(removed_colors)} "
                                if removed_ranks:
                                    diff_summary += f"Eliminated ranks: {list(removed_ranks)} "
                                diff_summary += f"({len(after_colors)}×{len(after_ranks)} possibilities remaining)\n"
                    
                    # For probabilistic variant - check distribution changes
                    elif 'color_distribution' in before_card:
                        before_color_dist = before_card['color_distribution']
                        after_color_dist = after_card['color_distribution']
                        before_rank_dist = before_card['rank_distribution']
                        after_rank_dist = after_card['rank_distribution']
                        
                        changed = False
                        changes = []
                        
                        # Check for certainty changes (0.0 or 1.0)
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
                            diff_summary += f"  {card_id}: {', '.join(changes)}\n"
        
        # Check Teammate beliefs
        if 'Teammate_Hand_Beliefs' in before and 'Teammate_Hand_Beliefs' in after:
            for player_hand in before['Teammate_Hand_Beliefs']:
                if player_hand in after['Teammate_Hand_Beliefs']:
                    before_hand = before['Teammate_Hand_Beliefs'][player_hand]
                    after_hand = after['Teammate_Hand_Beliefs'][player_hand]
                    
                    for card_id in before_hand:
                        if card_id in after_hand:
                            if 'p2_belief' in before_hand[card_id] and 'p2_belief' in after_hand[card_id]:
                                before_belief = before_hand[card_id]['p2_belief']
                                after_belief = after_hand[card_id]['p2_belief']
                                
                                # Check for changes in teammate beliefs
                                if before_belief != after_belief:
                                    diff_summary += f"  {player_hand}_{card_id}: Teammate belief updated\n"
        
        if diff_summary == "BELIEF CHANGES:\n":
            diff_summary += "  No significant changes detected\n"
        
        return diff_summary
    
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
            # Generic code fence without language hint
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
    
    def _get_belief_template(self) -> Dict:
        """Get a template for the belief graph structure that matches the current format."""
        # Return a copy of the current belief graph structure as the template
        return self.belief_graph
    
    def _detect_events(self, curr_obs: Dict[str, Any]) -> List[Dict]:
        """Detect ALL game events using the game state tracker."""
        tracker = get_game_state_tracker()
        events = []

        last_moves = curr_obs.get('last_moves', [])
        if not last_moves:
            return events

        # Update tracker with ALL observable state
        tracker.update_visible_cards(
            self.my_player_number,
            curr_obs.get('observed_hands', [])
        )

        if 'card_knowledge' in curr_obs and len(curr_obs['card_knowledge']) > 0:
            tracker.update_card_knowledge(
                self.my_player_number,
                curr_obs['card_knowledge'][0]
            )

        tracker.update_game_state(
            curr_obs.get('fireworks', {}),
            curr_obs.get('life_tokens', 3),
            curr_obs.get('information_tokens', 8),
            curr_obs.get('deck_size', 50)
        )

        num_players = curr_obs['num_players']

        self.logger.log_debug("EVENT_DETECT_TRACKER", f"Tracker state: {tracker.get_state_summary()}")

        # Process moves and register with tracker
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue
            self._seen_move_ids.add(move_id)

            move_data = last_move['move']
            action_type = move_data.get('action_type')

            if last_move['player'] < 0:
                continue

            acting_player_offset = last_move['player']
            acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players

            self.logger.log_debug(
                "EVENT_DETECT_TRACKER",
                f"Processing {action_type}: my_player={self.my_player_number}, "
                f"acting_offset={acting_player_offset}, acting_abs={acting_player_absolute}"
            )

            if action_type == 'REVEAL_COLOR':
                target_offset = move_data['target_offset']
                color = move_data['color']

                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                self.logger.log_debug(
                    "EVENT_DETECT_TRACKER",
                    f"REVEAL_COLOR: giver_abs={acting_player_absolute} (P{acting_player_absolute+1}), "
                    f"recipient_abs={target_player_absolute} (P{target_player_absolute+1}), color={color}"
                )

                card_indices = tracker.find_matching_cards_for_hint(
                    target_player_absolute, 'color', color
                )

                self.logger.log_debug(
                    "EVENT_DETECT_TRACKER",
                    f"Tracker found matching cards: {card_indices} in P{target_player_absolute+1}'s hand"
                )

                tracker.register_hint(
                    acting_player_absolute,
                    target_player_absolute,
                    'color',
                    color,
                    card_indices,
                    move_id
                )

                events.append({
                    'clue_giver': f"P{acting_player_absolute + 1}",
                    'clue_recipient': f"P{target_player_absolute + 1}",
                    'clue_type': 'color',
                    'value': color,
                    'card_indices': card_indices
                })

            elif action_type == 'REVEAL_RANK':
                target_offset = move_data['target_offset']
                rank = move_data['rank']

                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                self.logger.log_debug(
                    "EVENT_DETECT_TRACKER",
                    f"REVEAL_RANK: giver_abs={acting_player_absolute} (P{acting_player_absolute+1}), "
                    f"recipient_abs={target_player_absolute} (P{target_player_absolute+1}), rank={rank}"
                )

                card_indices = tracker.find_matching_cards_for_hint(
                    target_player_absolute, 'rank', rank
                )

                self.logger.log_debug(
                    "EVENT_DETECT_TRACKER",
                    f"Tracker found matching cards: {card_indices} in P{target_player_absolute+1}'s hand"
                )

                tracker.register_hint(
                    acting_player_absolute,
                    target_player_absolute,
                    'rank',
                    rank,
                    card_indices,
                    move_id
                )

                events.append({
                    'clue_giver': f"P{acting_player_absolute + 1}",
                    'clue_recipient': f"P{target_player_absolute + 1}",
                    'clue_type': 'rank',
                    'value': rank,
                    'card_indices': card_indices
                })

            elif action_type == 'PLAY':
                tracker.register_action(
                    acting_player_absolute,
                    'PLAY',
                    move_data.get('card_index', -1),
                    move_id
                )

                events.append({
                    'type': 'play',
                    'player': f"P{acting_player_absolute + 1}",
                    'card_index': move_data.get('card_index', -1),
                    'success': True
                })

            elif action_type == 'DISCARD':
                tracker.register_action(
                    acting_player_absolute,
                    'DISCARD',
                    move_data.get('card_index', -1),
                    move_id
                )

                events.append({
                    'type': 'discard',
                    'player': f"P{acting_player_absolute + 1}",
                    'card_index': move_data.get('card_index', -1)
                })

        self.logger.log_debug("EVENT_DETECT_TRACKER", f"Detected {len(events)} events this step")
        return events
    
    
    def _find_matching_cards(self, observation: Dict[str, Any], target_offset: int, clue_type: str, value) -> list:
        """Find which card indices match the given clue using the game state tracker.

        DEPRECATED: This method is kept for compatibility but now delegates to the tracker.
        Use tracker.find_matching_cards_for_hint() directly instead.
        """
        tracker = get_game_state_tracker()
        num_players = observation['num_players']

        target_player_absolute = (self.my_player_number + target_offset) % num_players

        self.logger.log_debug(
            "CARD_MATCHING_TRACKER",
            f"Finding cards for {clue_type}={value}, target_offset={target_offset}, "
            f"target_abs={target_player_absolute}"
        )

        matching_indices = tracker.find_matching_cards_for_hint(
            target_player_absolute, clue_type, value
        )

        self.logger.log_debug(
            "CARD_MATCHING_TRACKER",
            f"Tracker returned matching indices: {matching_indices}"
        )

        return matching_indices


class BeliefGraphProbabilisticAgent(GeminiAgent):
    """Variation 2: The Belief Graph (Probabilistic)
    
    Tracks probability distributions instead of sets.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.variant = 'probabilistic'
        self.agent_id = f"BG_probabilistic_{self.agent_id}"
        self.logger = AgentLogger(self.agent_id)
        self.prompt_manager = PromptManager()
        self.belief_graph = {}
        # 使用已处理 move 的唯一哈希集合进行去重，避免因历史长度相同而漏检
        self._seen_move_ids: set[str] = set()
        # 追踪上一帧的 observed_hands 来检测新牌
        self._previous_observed_hands = None
    
    def reset(self, config):
        """Reset and initialize probabilistic belief graph."""
        super().reset(config)

        # We need to determine our player number first
        self.my_player_number = None  # Will be set on first observation

        # Initialize with uniform distributions
        uniform_color = {"red": 0.2, "blue": 0.2, "green": 0.2, "white": 0.2, "yellow": 0.2}
        uniform_rank = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}

        self.belief_graph = {
            "GameState": {
                "clues": 8,
                "life": 3,
                "deck_size": 50
            },
            "My_Hand_Beliefs": {},
            "Teammate_Hand_Beliefs": {}
        }

        # Store uniform distributions for later use
        self.uniform_color = uniform_color
        self.uniform_rank = uniform_rank

        # 清空已处理 move 集合
        self._seen_move_ids.clear()
        # 清空上一帧 observed_hands
        self._previous_observed_hands = None

        # Initialize game state tracker
        tracker = get_game_state_tracker()
        num_players = config.get('players', 2)
        tracker.reset(num_players)

    # ------------------------------------------------------------------
    # New helper: Update game state tracker at the earliest possible point
    # ------------------------------------------------------------------
    def _update_game_state_tracker_early(self, observation: Dict[str, Any]):
        """Update game state tracker with the latest observation IMMEDIATELY.

        This is called at the very start of every act() call to ensure the tracker
        has the most current game state before any belief updates or processing.
        """
        tracker = get_game_state_tracker()

        self.logger.log_info("EARLY_UPDATE", "🔄 Updating game state tracker with latest observation")

        # Update visible cards immediately
        tracker.update_visible_cards(
            self.my_player_number,
            observation.get('observed_hands', [])
        )

        # Update card knowledge immediately
        if 'card_knowledge' in observation and len(observation['card_knowledge']) > 0:
            tracker.update_card_knowledge(
                self.my_player_number,
                observation['card_knowledge'][0]
            )

        # Update game state immediately
        tracker.update_game_state(
            observation.get('fireworks', {}),
            observation.get('life_tokens', 3),
            observation.get('information_tokens', 8),
            observation.get('deck_size', 50)
        )

        # Log the updated tracker state
        tracker_state = tracker.get_state_summary()
        self.logger.log_info("EARLY_UPDATE_COMPLETE",
                           f"✅ Game state tracker updated: {tracker_state}")

        # Update belief graph immediately with fresh tracker data
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size

        self.logger.log_info("BELIEF_SYNC", f"🔄 Belief graph synced: clues={tracker.clues}, lives={tracker.lives}, deck={tracker.deck_size}")

    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with probabilistic belief graph."""
        # Determine our player number on first observation
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True

        # CRITICAL: Update game state tracker IMMEDIATELY at the start of every act call
        # This ensures the tracker has the latest possible information before any processing
        self._update_game_state_tracker_early(observation)

        # CRITICAL: Process belief updates IMMEDIATELY after any hints
        if observation.get('last_moves'):
            # Update tracker and process hints IMMEDIATELY
            self._process_immediate_updates(observation)

        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None

        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
        # Update GameState with current fireworks from tracker
        tracker = get_game_state_tracker()
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'probabilistic'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()

        self.logger.log_info("BELIEF_STATE", f"Probabilistic graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))

        # Let parent handle all the action logic
        return super().act(augmented_observation)
    
    def _format_belief_graph_natural_language(self) -> str:
        """Convert probabilistic belief graph to natural language."""
        nl_description = "## BELIEF GRAPH ANALYSIS (PROBABILISTIC VARIANT)\n\n"
        
        # Game state
        gs = self.belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"
        
        # My hand beliefs with probabilities
        nl_description += "**MY HAND BELIEFS (with probabilities):**\n"
        for card_id, beliefs in self.belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            color_dist = beliefs['color_distribution']
            rank_dist = beliefs['rank_distribution']
            
            # Find highest probability color and rank
            max_color = max(color_dist, key=color_dist.get)
            max_color_prob = color_dist[max_color]
            max_rank = max(rank_dist, key=rank_dist.get)
            max_rank_prob = rank_dist[max_rank]
            
            if max_color_prob == 1.0 and max_rank_prob == 1.0:
                nl_description += f"- Card {card_num}: CERTAIN {max_color.upper()} {max_rank} (100% confidence)\n"
            elif max_color_prob == 1.0:
                nl_description += f"- Card {card_num}: CERTAIN color {max_color.upper()}, likely rank {max_rank} ({max_rank_prob*100:.0f}%)\n"
            elif max_rank_prob == 1.0:
                nl_description += f"- Card {card_num}: CERTAIN rank {max_rank}, likely {max_color} ({max_color_prob*100:.0f}%)\n"
            else:
                nl_description += f"- Card {card_num}: Maybe {max_color} {max_rank} ({max_color_prob*100:.0f}% color, {max_rank_prob*100:.0f}% rank)\n"
        
        # Teammate beliefs with probabilities
        nl_description += "\n**TEAMMATE KNOWLEDGE MODEL (probabilistic):**\n"
        for player_hand, hand_data in self.belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            nl_description += f"\nPlayer {player_num}'s cards (probability distributions):\n"
            
            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')
                belief = card_data.get(f'p{player_num}_belief', {})
                color_dist = belief.get('color_distribution', {})
                rank_dist = belief.get('rank_distribution', {})
                
                # Find certainties
                certain_color = None
                certain_rank = None
                for c, p in color_dist.items():
                    if p == 1.0:
                        certain_color = c
                for r, p in rank_dist.items():
                    if p == 1.0:
                        certain_rank = r
                
                if certain_color and certain_rank:
                    nl_description += f"  - Card {card_num}: I see {actual}, they're CERTAIN it's {certain_color.upper()} {certain_rank}\n"
                elif certain_color:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know COLOR {certain_color.upper()} (100%)\n"
                elif certain_rank:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know RANK {certain_rank} (100%)\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, they have uncertainty\n"
        
        return nl_description
    
    def _determine_player_number(self, observation: Dict[str, Any]):
        """Determine which player we are based on the observation.
        
        The key insight: current_player_offset tells us how many seats away 
        the current player is from us (the observer).
        
        So: my_player_number = (current_player - current_player_offset) % num_players
        """
        current_player = observation['current_player']
        current_player_offset = observation['current_player_offset']
        num_players = observation['num_players']
        
        # Calculate our player number
        self.my_player_number = (current_player - current_player_offset) % num_players
        self.logger.log_info("PLAYER_IDENTITY", 
                           f"I am Player {self.my_player_number + 1} " +
                           f"(current_player={current_player}, offset={current_player_offset})")
    
    def _initialize_belief_graph_with_player_number(self, observation: Dict[str, Any]):
        """Initialize belief graph with correct player numbers."""
        my_player_id = self.my_player_number + 1  # Convert to 1-indexed
        num_players = observation['num_players']
        
        # Initialize my hand beliefs with probability distributions
        for i in range(5):  # Assuming 5 cards per hand
            self.belief_graph["My_Hand_Beliefs"][f"P{my_player_id}_Card{i+1}"] = {
                "color_distribution": self.uniform_color.copy(),
                "rank_distribution": self.uniform_rank.copy()
            }
        
        # Initialize teammate beliefs
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"] = {}
                for card in range(5):
                    self.belief_graph["Teammate_Hand_Beliefs"][f"P{player_id}_Hand"][f"P{player_id}_Card{card+1}"] = {
                        "actual_card_I_see": "Unknown",
                        f"p{player_id}_belief": {
                            "color_distribution": self.uniform_color.copy(),
                            "rank_distribution": self.uniform_rank.copy()
                        }
                    }
        
        # Initialize visible cards
        self._initialize_visible_cards(observation)
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Update with visible cards."""
        num_players = observation['num_players']
        for offset, hand in enumerate(observation['observed_hands']):
            if offset == 0:
                continue  # Skip my own hand
            
            # Calculate actual player number from offset
            actual_player_num = (self.my_player_number + offset) % num_players
            player_id = f"P{actual_player_num + 1}"
            
            if f"{player_id}_Hand" in self.belief_graph["Teammate_Hand_Beliefs"]:
                for card_idx, card in enumerate(hand):
                    if card.get('color') and card.get('rank') is not None:
                        card_key = f"{player_id}_Card{card_idx + 1}"
                        color = card['color']
                        rank = card['rank'] + 1 if card['rank'] >= 0 else '?'
                        self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"][card_key]["actual_card_I_see"] = f"{color} {rank}"
    
    def _update_beliefs_via_llm(self, observation: Dict[str, Any]):
        """Update probabilistic beliefs via LLM."""
        # Detect all events that happened
        events = self._detect_events(observation)
        if not events:
            return
        
        # Process each event one by one
        for i, event in enumerate(events):
            self.logger.log_info("EVENT_OUT", f"Detected event {i+1}/{len(events)}: {json.dumps(event, indent=2)}")
            self._process_single_event_probabilistic(event)
    
    def _process_single_event_probabilistic(self, event: Dict[str, Any]):
        """Process a single event through the LLM for probabilistic beliefs."""
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        # Use intelligent probabilistic prompt
        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'probabilistic')

        # Log the update prompt
        self.logger.log_info("BELIEF_UPDATE_PROMPT", f"Sending probabilistic update to LLM: {len(prompt)} chars")
        self.logger.log_debug("BELIEF_UPDATE_PROMPT_DETAIL", prompt)

        try:
            # First get the full reasoning response
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2}
            )

            # Log the full reasoning
            self.logger.log_info("BELIEF_UPDATE_RESPONSE", f"LLM reasoning response: {len(reasoning_response.text)} chars")
            self.logger.log_debug("BELIEF_UPDATE_COT_REASONING", reasoning_response.text)

            # Extract JSON from response - look for the main JSON block
            response_text = reasoning_response.text

            # Find ```json markers or look for the main JSON block
            if '```json' in response_text:
                json_start = response_text.find('```json') + 7
                json_end = response_text.find('```', json_start)
                if json_end == -1:
                    json_end = len(response_text)
                json_text = response_text[json_start:json_end].strip()
            else:
                # Find the largest JSON object by looking for complete braces
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
                            # Check if this looks like a complete JSON by trying to parse
                            try:
                                potential_json = response_text[json_start:json_end]
                                json.loads(potential_json)
                                json_text = potential_json
                                break
                            except:
                                continue

                if json_start == -1 or json_end == -1:
                    raise ValueError("No valid JSON found in LLM response")

            self.logger.log_debug("BELIEF_UPDATE_JSON_EXTRACTED", json_text)

            # Store previous belief state for comparison
            previous_belief = json.dumps(self.belief_graph, indent=2)
            self.logger.log_debug("BELIEF_UPDATE_BEFORE", previous_belief)

            # Update with new belief state
            new_belief_graph = json.loads(json_text)

            # Generate and log diff BEFORE updating
            diff_summary = self._generate_belief_diff(self.belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            # Now update the belief graph
            self.belief_graph = new_belief_graph

            # Log the updated belief state
            updated_belief = json.dumps(self.belief_graph, indent=2)
            self.logger.log_debug("BELIEF_UPDATE_AFTER", updated_belief)
            self.logger.log_info("LLM_UPDATE", "Probabilistic belief graph updated via LLM")

        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update probabilistic beliefs: {e}")
    
    # Use the base class _detect_events and _find_matching_cards methods

    def _process_immediate_updates(self, observation: Dict[str, Any]):
        """Process IMMEDIATE belief updates after any game actions."""
        tracker = get_game_state_tracker()
        num_players = observation['num_players']

        # The tracker is ALREADY updated in _update_game_state_tracker_early()
        # No need to update it again here - use the current state

        # NOW process hint events with updated tracker data
        last_moves = observation.get('last_moves', [])
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue

            move_data = last_move.get('move', {})
            action_type = move_data.get('action_type')

            if last_move.get('player', -1) < 0:
                continue

            # Process hint events immediately
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                acting_player_offset = last_move['player']
                acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players

                target_offset = move_data['target_offset']
                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                if action_type == 'REVEAL_COLOR':
                    color = move_data['color']
                    # Tracker has correct data to find matching cards
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
                    # Tracker has correct data to find matching cards
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

                # Process the hint event IMMEDIATELY
                self.logger.log_info("IMMEDIATE_UPDATE", f"Processing immediate probabilistic belief update: {json.dumps(event, indent=2)}")
                self._process_single_event_probabilistic(event)
                self._seen_move_ids.add(move_id)

    def _generate_belief_diff(self, before: Dict, after: Dict) -> str:
        """Generate a human-readable diff of belief changes."""
        diff_summary = "BELIEF CHANGES:\n"
        
        # Check My_Hand_Beliefs for changes
        if 'My_Hand_Beliefs' in before and 'My_Hand_Beliefs' in after:
            for card_id in before['My_Hand_Beliefs']:
                if card_id in after['My_Hand_Beliefs']:
                    before_card = before['My_Hand_Beliefs'][card_id]
                    after_card = after['My_Hand_Beliefs'][card_id]
                    
                    # For certainty variant - check possible values
                    if 'possible_colors' in before_card:
                        before_colors = set(before_card['possible_colors'])
                        after_colors = set(after_card['possible_colors'])
                        before_ranks = set(before_card['possible_ranks'])
                        after_ranks = set(after_card['possible_ranks'])
                        
                        if before_colors != after_colors or before_ranks != after_ranks:
                            removed_colors = before_colors - after_colors
                            removed_ranks = before_ranks - after_ranks

                            if removed_colors or removed_ranks:
                                diff_summary += f"  {card_id}: "
                                if removed_colors:
                                    diff_summary += f"Eliminated colors: {list(removed_colors)} "
                                if removed_ranks:
                                    diff_summary += f"Eliminated ranks: {list(removed_ranks)} "
                                diff_summary += f"({len(after_colors)}×{len(after_ranks)} possibilities remaining)\n"
                    
                    # For probabilistic variant - check distribution changes
                    elif 'color_distribution' in before_card:
                        before_color_dist = before_card['color_distribution']
                        after_color_dist = after_card['color_distribution']
                        before_rank_dist = before_card['rank_distribution']
                        after_rank_dist = after_card['rank_distribution']
                        
                        changed = False
                        changes = []
                        
                        # Check for certainty changes (0.0 or 1.0)
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
                            diff_summary += f"  {card_id}: {', '.join(changes)}\n"
        
        # Check Teammate beliefs
        if 'Teammate_Hand_Beliefs' in before and 'Teammate_Hand_Beliefs' in after:
            for player_hand in before['Teammate_Hand_Beliefs']:
                if player_hand in after['Teammate_Hand_Beliefs']:
                    before_hand = before['Teammate_Hand_Beliefs'][player_hand]
                    after_hand = after['Teammate_Hand_Beliefs'][player_hand]
                    
                    for card_id in before_hand:
                        if card_id in after_hand:
                            if 'p2_belief' in before_hand[card_id] and 'p2_belief' in after_hand[card_id]:
                                before_belief = before_hand[card_id]['p2_belief']
                                after_belief = after_hand[card_id]['p2_belief']
                                
                                # Check for changes in teammate beliefs
                                if before_belief != after_belief:
                                    diff_summary += f"  {player_hand}_{card_id}: Teammate belief updated\n"
        
        if diff_summary == "BELIEF CHANGES:\n":
            diff_summary += "  No significant changes detected\n"
        
        return diff_summary


    def _get_belief_template(self) -> Dict:
        """Get a template for the belief graph structure that matches the current format."""
        # Return a copy of the current belief graph structure as the template
        return self.belief_graph


class BeliefGraphToMAgent(GeminiAgent):
    """Variation 3: The "Player Read" Graph (Recursive ToM)
    
    Adds Theory of Mind layer on top of probabilistic beliefs.
    """
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.variant = 'theory_of_mind'
        self.agent_id = f"BG_tom_{self.agent_id}"
        self.logger = AgentLogger(self.agent_id)
        self.prompt_manager = PromptManager()
        self.belief_graph = {}
        # 使用已处理 move 的唯一哈希集合进行去重，避免因历史长度相同而漏检
        self._seen_move_ids: set[str] = set()
        # 追踪上一帧的 observed_hands 来检测新牌
        self._previous_observed_hands = None
    
    def reset(self, config):
        """Reset with ToM layer."""
        super().reset(config)

        # We need to determine our player number first
        self.my_player_number = None  # Will be set on first observation

        uniform_color = {"red": 0.2, "blue": 0.2, "green": 0.2, "white": 0.2, "yellow": 0.2}
        uniform_rank = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}

        self.belief_graph = {
            "GameState": {
                "clues": 8,
                "life": 3,
                "deck_size": 50
            },
            "My_Hand_Beliefs": {},
            "Teammate_Hand_Beliefs": {},
            "ToM_Layer": {  # Theory of Mind layer as specified
                "Teammates": {},
                "Team_Focus": {
                    "focus_distribution": {
                        "PLAY_ANYTHING": 0.9,
                        "SAVE_FIVES": 0.05,
                        "PLAY_ONES": 0.05
                    }
                }
            }
        }

        # Store uniform distributions for later use
        self.uniform_color = uniform_color
        self.uniform_rank = uniform_rank

        # 清空已处理 move 集合
        self._seen_move_ids.clear()
        # 清空上一帧 observed_hands
        self._previous_observed_hands = None

        # Initialize game state tracker
        tracker = get_game_state_tracker()
        num_players = config.get('players', 2)
        tracker.reset(num_players)

    # ------------------------------------------------------------------
    # New helper: Update game state tracker at the earliest possible point
    # ------------------------------------------------------------------
    def _update_game_state_tracker_early(self, observation: Dict[str, Any]):
        """Update game state tracker with the latest observation IMMEDIATELY.

        This is called at the very start of every act() call to ensure the tracker
        has the most current game state before any belief updates or processing.
        """
        tracker = get_game_state_tracker()

        self.logger.log_info("EARLY_UPDATE", "🔄 Updating game state tracker with latest observation")

        # Update visible cards immediately
        tracker.update_visible_cards(
            self.my_player_number,
            observation.get('observed_hands', [])
        )

        # Update card knowledge immediately
        if 'card_knowledge' in observation and len(observation['card_knowledge']) > 0:
            tracker.update_card_knowledge(
                self.my_player_number,
                observation['card_knowledge'][0]
            )

        # Update game state immediately
        tracker.update_game_state(
            observation.get('fireworks', {}),
            observation.get('life_tokens', 3),
            observation.get('information_tokens', 8),
            observation.get('deck_size', 50)
        )

        # Log the updated tracker state
        tracker_state = tracker.get_state_summary()
        self.logger.log_info("EARLY_UPDATE_COMPLETE",
                           f"✅ Game state tracker updated: {tracker_state}")

        # Update belief graph immediately with fresh tracker data
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size

        self.logger.log_info("BELIEF_SYNC", f"🔄 Belief graph synced: clues={tracker.clues}, lives={tracker.lives}, deck={tracker.deck_size}")

    def _initialize_belief_graph_with_player_number(self, observation: Dict[str, Any]):
        """Initialize belief graph with ToM layer."""
        # First, call the parent class to initialize basic beliefs
        super()._initialize_belief_graph_with_player_number(observation)
        
        # Then add ToM models for teammates
        num_players = observation['num_players']
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                self.belief_graph["ToM_Layer"]["Teammates"][f"P{player_id}"] = {
                    "inferred_skill": 0.5,
                    "play_aggressiveness": 0.5,
                    "hint_quality": 0.5
                }
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with ToM-enhanced belief graph."""
        # Determine our player number on first observation
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True

        # CRITICAL: Update game state tracker IMMEDIATELY at the start of every act call
        # This ensures the tracker has the latest possible information before any processing
        self._update_game_state_tracker_early(observation)

        # CRITICAL: Process belief updates IMMEDIATELY after any hints
        if observation.get('last_moves'):
            # Update tracker and process hints IMMEDIATELY
            self._process_immediate_updates(observation)

        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None

        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
        # Update GameState with current fireworks from tracker
        tracker = get_game_state_tracker()
        self.belief_graph['GameState']['fireworks'] = tracker.get_fireworks()
        self.belief_graph['GameState']['clues'] = tracker.clues
        self.belief_graph['GameState']['life'] = tracker.lives
        self.belief_graph['GameState']['deck_size'] = tracker.deck_size
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'theory_of_mind'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()

        self.logger.log_info("BELIEF_STATE", f"ToM graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))

        # Let parent handle all the action logic
        return super().act(augmented_observation)
    
    def _format_belief_graph_natural_language(self) -> str:
        """Convert ToM belief graph to natural language with player modeling."""
        nl_description = "## BELIEF GRAPH ANALYSIS (THEORY OF MIND VARIANT)\n\n"
        
        # Game state
        gs = self.belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"
        
        # ToM Layer - Team dynamics
        tom = self.belief_graph['ToM_Layer']
        nl_description += "**TEAM DYNAMICS ANALYSIS:**\n"
        
        # Team focus
        team_focus = tom['Team_Focus']['focus_distribution']
        max_focus = max(team_focus, key=team_focus.get)
        focus_prob = team_focus[max_focus]
        nl_description += f"- Team Strategy: {max_focus} ({focus_prob*100:.0f}% confidence)\n"
        
        # Teammate profiles
        nl_description += "\n**TEAMMATE PROFILES:**\n"
        for player, profile in tom['Teammates'].items():
            skill = profile['inferred_skill']
            aggr = profile.get('play_aggressiveness', 0.5)
            hint_q = profile.get('hint_quality', 0.5)
            
            skill_desc = "expert" if skill > 0.8 else "competent" if skill > 0.6 else "average" if skill > 0.4 else "novice"
            aggr_desc = "aggressive" if aggr > 0.7 else "balanced" if aggr > 0.3 else "conservative"
            hint_desc = "excellent hints" if hint_q > 0.8 else "good hints" if hint_q > 0.6 else "average hints" if hint_q > 0.4 else "poor hints"
            
            nl_description += f"- {player}: {skill_desc} player (skill: {skill:.1f}), {aggr_desc} style, {hint_desc}\n"
        
        # Regular probabilistic beliefs
        nl_description += "\n**CARD BELIEFS (probabilistic base):**\n"
        
        # My hand
        nl_description += "\nMy Hand:\n"
        for card_id, beliefs in self.belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            color_dist = beliefs['color_distribution']
            rank_dist = beliefs['rank_distribution']
            
            max_color = max(color_dist, key=color_dist.get)
            max_rank = max(rank_dist, key=rank_dist.get)
            
            nl_description += f"- Card {card_num}: Likely {max_color} {max_rank}\n"
        
        # Teammate hands with ToM context
        nl_description += "\n**TEAMMATE HANDS (with mental models):**\n"
        for player_hand, hand_data in self.belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            player_key = f"P{player_num}"
            
            # Get player profile for context
            if player_key in tom['Teammates']:
                skill = tom['Teammates'][player_key]['inferred_skill']
                nl_description += f"\nPlayer {player_num} (skill: {skill:.1f}):\n"
            else:
                nl_description += f"\nPlayer {player_num}:\n"
            
            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')
                nl_description += f"  - Card {card_num}: I see {actual}\n"
        
        return nl_description
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Initialize visible cards."""
        for player_idx, hand in enumerate(observation['observed_hands']):
            if player_idx == 0:
                continue
            
            player_id = f"P{player_idx + 1}"
            if f"{player_id}_Hand" in self.belief_graph["Teammate_Hand_Beliefs"]:
                for card_idx, card in enumerate(hand):
                    if card.get('color') and card.get('rank') is not None:
                        card_key = f"{player_id}_Card{card_idx + 1}"
                        color = card['color']
                        rank = card['rank'] + 1 if card['rank'] >= 0 else '?'
                        self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"][card_key]["actual_card_I_see"] = f"{color} {rank}"
    
    def _update_beliefs_with_tom(self, observation: Dict[str, Any]):
        """Update with Theory of Mind reasoning."""
        # Detect all events that happened
        events = self._detect_events(observation)
        if not events:
            return
        
        # Process each event one by one
        for i, event in enumerate(events):
            self.logger.log_info("EVENT_OUT", f"Detected event {i+1}/{len(events)}: {json.dumps(event, indent=2)}")
            self._process_single_event_tom(event)
    
    def _process_single_event_tom(self, event: Dict[str, Any]):
        """Process a single event through the LLM with ToM reasoning."""
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        # Use intelligent ToM prompt
        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'theory_of_mind')

        # Log the ToM update prompt
        self.logger.log_info("TOM_UPDATE_PROMPT", f"Sending ToM update to LLM: {len(prompt)} chars")
        self.logger.log_debug("TOM_UPDATE_PROMPT_DETAIL", prompt)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'response_mime_type': 'application/json', 'temperature': 0.2}
            )

            # Log the response
            self.logger.log_info("TOM_UPDATE_RESPONSE", f"LLM response: {len(response.text)} chars")
            self.logger.log_debug("TOM_UPDATE_RESPONSE_DETAIL", response.text)

            # Store previous belief state for comparison
            previous_belief_graph = self.belief_graph.copy()

            # Update with new belief state
            new_belief_graph = json.loads(response.text)

            # Generate and log diff BEFORE updating
            diff_summary = self._generate_belief_diff(previous_belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            # Now update the belief graph
            self.belief_graph = new_belief_graph

            # Log ToM insights
            if 'ToM_Layer' in self.belief_graph:
                self.logger.log_info("TOM_UPDATE", f"Team focus: {self.belief_graph['ToM_Layer']['Team_Focus']}")
                self.logger.log_info("LLM_UPDATE", "ToM belief graph updated via LLM")

        except Exception as e:
            self.logger.log_error("TOM_ERROR", f"Failed to update ToM beliefs: {e}")

    def _process_immediate_updates(self, observation: Dict[str, Any]):
        """Process IMMEDIATE belief updates after any game actions."""
        tracker = get_game_state_tracker()
        num_players = observation['num_players']

        # The tracker is ALREADY updated in _update_game_state_tracker_early()
        # No need to update it again here - use the current state

        # NOW process hint events with updated tracker data
        last_moves = observation.get('last_moves', [])
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue

            move_data = last_move.get('move', {})
            action_type = move_data.get('action_type')

            if last_move.get('player', -1) < 0:
                continue

            # Process hint events immediately
            if action_type in ['REVEAL_COLOR', 'REVEAL_RANK']:
                acting_player_offset = last_move['player']
                acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players

                target_offset = move_data['target_offset']
                target_player_absolute = (acting_player_absolute + target_offset) % num_players

                if action_type == 'REVEAL_COLOR':
                    color = move_data['color']
                    # Tracker has correct data to find matching cards
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
                    # Tracker has correct data to find matching cards
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

                # Process the hint event IMMEDIATELY
                self.logger.log_info("IMMEDIATE_UPDATE", f"Processing immediate ToM belief update: {json.dumps(event, indent=2)}")
                self._process_single_event_tom(event)
                self._seen_move_ids.add(move_id)
    
    def _detect_events(self, curr_obs: Dict[str, Any]) -> List[Dict]:
        """Detect game events by comparing observations."""
        # Get the last move from history if available
        last_moves = curr_obs['last_moves']
        if not last_moves:
            return None
            
        # Look at the most recent move
        last_move = last_moves[-1]
        move_data = last_move['move']
        action_type = move_data.get('action_type')
        
        # Check for different action types
        if action_type == 'REVEAL_COLOR':
            target_offset = move_data['target_offset']
            color = move_data['color']
            
            return {
                'clue_giver': f"P{last_move['player'] + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'color',
                'value': color,
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'color', color)
            }
            
        elif action_type == 'REVEAL_RANK':
            target_offset = move_data['target_offset']
            rank = move_data['rank']
            
            return {
                'clue_giver': f"P{last_move['player'] + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'rank',
                'value': rank,  # rank is already 1-indexed from LLM/game, keep it for belief graph
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'rank', rank)
            }
        
        return None
    
    def _generate_belief_diff(self, before: Dict, after: Dict) -> str:
        """Generate a human-readable diff of belief changes."""
        diff_summary = "BELIEF CHANGES:\n"
        
        # Check My_Hand_Beliefs for changes
        if 'My_Hand_Beliefs' in before and 'My_Hand_Beliefs' in after:
            for card_id in before['My_Hand_Beliefs']:
                if card_id in after['My_Hand_Beliefs']:
                    before_card = before['My_Hand_Beliefs'][card_id]
                    after_card = after['My_Hand_Beliefs'][card_id]
                    
                    # For certainty variant - check possible values
                    if 'possible_colors' in before_card:
                        before_colors = set(before_card['possible_colors'])
                        after_colors = set(after_card['possible_colors'])
                        before_ranks = set(before_card['possible_ranks'])
                        after_ranks = set(after_card['possible_ranks'])
                        
                        if before_colors != after_colors or before_ranks != after_ranks:
                            removed_colors = before_colors - after_colors
                            removed_ranks = before_ranks - after_ranks

                            if removed_colors or removed_ranks:
                                diff_summary += f"  {card_id}: "
                                if removed_colors:
                                    diff_summary += f"Eliminated colors: {list(removed_colors)} "
                                if removed_ranks:
                                    diff_summary += f"Eliminated ranks: {list(removed_ranks)} "
                                diff_summary += f"({len(after_colors)}×{len(after_ranks)} possibilities remaining)\n"
                    
                    # For probabilistic variant - check distribution changes
                    elif 'color_distribution' in before_card:
                        before_color_dist = before_card['color_distribution']
                        after_color_dist = after_card['color_distribution']
                        before_rank_dist = before_card['rank_distribution']
                        after_rank_dist = after_card['rank_distribution']
                        
                        changed = False
                        changes = []
                        
                        # Check for certainty changes (0.0 or 1.0)
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
                            diff_summary += f"  {card_id}: {', '.join(changes)}\n"
        
        # Check Teammate beliefs
        if 'Teammate_Hand_Beliefs' in before and 'Teammate_Hand_Beliefs' in after:
            for player_hand in before['Teammate_Hand_Beliefs']:
                if player_hand in after['Teammate_Hand_Beliefs']:
                    before_hand = before['Teammate_Hand_Beliefs'][player_hand]
                    after_hand = after['Teammate_Hand_Beliefs'][player_hand]
                    
                    for card_id in before_hand:
                        if card_id in after_hand:
                            # Check for changes in teammate beliefs (works for both certainty and probabilistic)
                            for key in before_hand[card_id]:
                                if key.startswith('p') and key.endswith('_belief'):
                                    before_belief = before_hand[card_id][key]
                                    after_belief = after_hand[card_id][key]
                                    
                                    if before_belief != after_belief:
                                        diff_summary += f"  {player_hand}_{card_id}: Teammate belief updated\n"
        
        # Check ToM Layer changes
        if 'ToM_Layer' in before and 'ToM_Layer' in after:
            before_tom = before['ToM_Layer']
            after_tom = after['ToM_Layer']
            
            # Check Team_Focus changes
            if 'Team_Focus' in before_tom and 'Team_Focus' in after_tom:
                before_focus = before_tom['Team_Focus']['focus_distribution']
                after_focus = after_tom['Team_Focus']['focus_distribution']
                
                for focus_type in before_focus:
                    if before_focus[focus_type] != after_focus[focus_type]:
                        diff_summary += f"  Team_Focus.{focus_type}: {before_focus[focus_type]:.2f} → {after_focus[focus_type]:.2f}\n"
            
            # Check Teammate skill changes
            if 'Teammates' in before_tom and 'Teammates' in after_tom:
                for player in before_tom['Teammates']:
                    if player in after_tom['Teammates']:
                        before_player = before_tom['Teammates'][player]
                        after_player = after_tom['Teammates'][player]
                        
                        for attribute in before_player:
                            if before_player[attribute] != after_player[attribute]:
                                diff_summary += f"  {player}.{attribute}: {before_player[attribute]:.2f} → {after_player[attribute]:.2f}\n"
        
        if diff_summary == "BELIEF CHANGES:\n":
            diff_summary += "  No significant changes detected\n"
        
        return diff_summary
