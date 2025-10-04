"""Belief graph scaffolding agents - EXACTLY as specified.

These are thin layers on GeminiAgent that maintain JSON belief structures
and pass them to the LLM for reasoning. No complex graph infrastructure needed.
"""

import json
from typing import Dict, Any, Optional, List

from ..agents.gemini_agent import GeminiAgent
from ..agents.agent_logger import AgentLogger


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
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with belief graph augmentation."""
        # Determine our player number on first observation
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True
        
        # Always keep teammate actual cards up to date
        self._refresh_visible_cards(observation)

        # Always update beliefs when observing, regardless of whose turn it is
        if observation.get('last_moves'):
            self._update_beliefs_via_llm(observation)
        
        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None
        
        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
        # Update GameState with current fireworks for accurate playability checking
        self.belief_graph['GameState']['fireworks'] = observation.get('fireworks', {})
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'certainty'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()
        
        # Log current belief state
        self.logger.log_info("BELIEF_STATE", f"Certainty graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))
        
        # Let parent handle all the action logic
        return super().act(augmented_observation)
    
    # ------------------------------------------------------------------
    # New helper: refresh teammate actual cards each observation to avoid
    # stale duplicates when hands change after plays/discards
    # ------------------------------------------------------------------
    def _refresh_visible_cards(self, observation):
        """Update actual_card_I_see fields for teammate hands every turn."""
        num_players = observation['num_players']
        self.logger.log_debug("CARD_REFRESH", "🔍 Starting card refresh check...")
        
        current_hands = observation.get('observed_hands', [])
        
        # Check for PLAY/DISCARD actions that cause hand shifts
        # When I play/discard, all cards to the right shift left, so we need to reset beliefs
        if 'last_moves' in observation:
            for move_entry in observation['last_moves']:
                player_offset = move_entry.get('player', -1)
                if player_offset < 0:
                    continue
                
                # Calculate absolute player number
                current_player = observation['current_player']
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
        
        # Handle TEAMMATE hands - we CAN see these change
        for offset, hand in enumerate(current_hands):
            if offset == 0:
                continue  # skip my own hand

            actual_player_num = (self.my_player_number + offset) % num_players
            player_id = f"P{actual_player_num + 1}"

            hand_key = f"{player_id}_Hand"
            if hand_key not in self.belief_graph.get("Teammate_Hand_Beliefs", {}):
                continue

            # Ensure correct number of card entries exists
            for card_idx, card in enumerate(hand):
                card_key = f"{player_id}_Card{card_idx + 1}"

                if card.get('color') is not None and card.get('rank') is not None:
                    color = card['color']
                    rank = card['rank'] + 1 if card['rank'] >= 0 else '?'
                    new_actual = f"{color} {rank}"
                    # Just update the actual card - don't reset belief here
                    # Belief shifting is handled by PLAY/DISCARD action detection
                    self.belief_graph["Teammate_Hand_Beliefs"][hand_key][card_key]["actual_card_I_see"] = new_actual
                else:
                    # Unknown card (face down after draw) → mark Unknown
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
        """Update belief graph with visible cards."""
        # Update teammate cards I can see
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
        # Get agent's actual player number (already determined)
        my_player_id = f"P{self.my_player_number + 1}"
        
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return
            
        # (3) Log belief graph before LLM update
        self.logger.log_info("BELIEF_BEFORE_UPDATE", f"Current belief graph: {json.dumps(self.belief_graph, indent=2)}")
            
        # Check if this is about our own hand or teammate's
        if event['clue_recipient'] == my_player_id:
            # Update our own beliefs
            clue_type = event.get('clue_type')
            clued_value = event.get('value')
            clued_indices = event.get('card_indices', [])
            
            prompt = f"""You are agent {my_player_id}. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event: {json.dumps(event)}

Your task is to update your model of your own hand beliefs (My_Hand_Beliefs) based on this clue event.

🔴 CRITICAL: NEGATIVE INFERENCE IS MANDATORY 🔴

⚠️ INDEXING: card_indices uses 0-BASED ARRAY INDEXING ⚠️
   - Index 0 = P2_Card1 (first card)
   - Index 1 = P2_Card2 (second card)
   - Index 2 = P2_Card3 (third card)
   - Index 3 = P2_Card4 (fourth card)
   - Index 4 = P2_Card5 (fifth card)

Step-by-step process you MUST follow:
1. Clued cards (0-indexed: {clued_indices}): Set possible_{clue_type}s to ONLY [{clued_value}]
2. Un-clued cards (indices 0-4 NOT in {clued_indices}): REMOVE {clued_value} from possible_{clue_type}s, keep others
3. EXCEPTION: If a card already has only 1 possibility, don't change it

CONCRETE EXAMPLE for this event:
Clue: {clue_type} = {clued_value}, targeting card_indices {clued_indices} (0-indexed)

For each card index (0-4):
- If index in {clued_indices}: possible_{clue_type}s = ["{clued_value}"]
- If index NOT in {clued_indices}: remove "{clued_value}" from possible_{clue_type}s

VERIFICATION CHECKLIST (check before returning):
✓ Did I correctly map card_indices to P2_Card# keys? (index 0 → P2_Card1, index 1 → P2_Card2, etc.)
✓ Clued cards (indices {clued_indices}) have ONLY [{clued_value}] in possible_{clue_type}s?
✓ Un-clued cards (indices 0-4 NOT in {clued_indices}) have {clued_value} REMOVED from possible_{clue_type}s?
✓ All other attributes preserved?

Return the complete updated JSON graph."""
        else:
            # Update teammate beliefs
            clue_type = event.get('clue_type')
            clued_value = event.get('value')
            clued_indices = event.get('card_indices', [])
            recipient_id = event['clue_recipient']

            prompt = f"""You are agent {my_player_id}. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event: {json.dumps(event)}

Your task is to update your model of {recipient_id}'s beliefs in Teammate_Hand_Beliefs.

🔴 CRITICAL: YOU MUST UPDATE THE BELIEF GRAPH 🔴

This clue was given to {recipient_id}. You need to update what {recipient_id} now believes about their own hand.
Navigate to: Teammate_Hand_Beliefs → {recipient_id}_Hand → {recipient_id}_Card# → {recipient_id.lower()}_belief

🔴 CRITICAL: NEGATIVE INFERENCE IS MANDATORY 🔴

⚠️ INDEXING: card_indices uses 0-BASED ARRAY INDEXING ⚠️
   - Index 0 = {recipient_id}_Card1 (first card)
   - Index 1 = {recipient_id}_Card2 (second card)
   - Index 2 = {recipient_id}_Card3 (third card)
   - Index 3 = {recipient_id}_Card4 (fourth card)
   - Index 4 = {recipient_id}_Card5 (fifth card)

Step-by-step process you MUST follow:
1. Identify which cards were clued (0-indexed): {event.get('card_indices', [])}
2. For CLUED cards at indices {event.get('card_indices', [])}:
   - Update {recipient_id}_Card#.{recipient_id.lower()}_belief.possible_{clue_type}s to ONLY [{clued_value}]
3. For UN-CLUED cards (all other indices 0-4):
   - Update {recipient_id}_Card#.{recipient_id.lower()}_belief: REMOVE {clued_value} from possible_{clue_type}s
   - Keep all other values unchanged

CONCRETE EXAMPLE FOR THIS EVENT:
Clue: {clue_type} = {clued_value}, targeting card_indices {clued_indices} (0-indexed)
Updating beliefs in: Teammate_Hand_Beliefs.{recipient_id}_Hand.{recipient_id}_Card#.{recipient_id.lower()}_belief

For each card index (0-4):
- If index in {clued_indices}: set {recipient_id.lower()}_belief.possible_{clue_type}s = ["{clued_value}"]
- If index NOT in {clued_indices}: remove "{clued_value}" from {recipient_id.lower()}_belief.possible_{clue_type}s

VERIFICATION CHECKLIST (check before returning):
✓ Did I correctly map card_indices to {recipient_id}_Card# keys? (index 0 → {recipient_id}_Card1, index 1 → {recipient_id}_Card2, etc.)
✓ Did I update the {recipient_id.lower()}_belief field (NOT the top level)?
✓ Did I set clued cards' {recipient_id.lower()}_belief.possible_{clue_type}s to ONLY ["{clued_value}"]?
✓ Did I remove "{clued_value}" from ALL un-clued cards' {recipient_id.lower()}_belief.possible_{clue_type}s?
✓ Did I preserve "actual_card_I_see" fields unchanged?
✓ Did I keep all OTHER possible values for un-clued cards?

Return the complete updated JSON graph."""
        
        # (4) Log prompt into LLM
        self.logger.log_info("PROMPT_TO_LLM", f"Prompt ({len(prompt)} chars):\n{prompt}")
        
        try:
            # Get the full reasoning response
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.7}
            )
            
            # (5) Log LLM's output
            self.logger.log_info("LLM_OUTPUT", f"LLM Response:\n{reasoning_response.text}")
            
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
                            card_num = card_id.split('Card')[1]
                            
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
                            card_num = card_id.split('Card')[1]
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
    
    def _detect_events(self, curr_obs: Dict[str, Any]) -> List[Dict]:
        """Detect ALL game events from observation - returns a list of events."""
        events: List[Dict] = []

        last_moves = curr_obs.get('last_moves', [])

        if not last_moves:
            return events

        new_move_count = 0

        # 逐条检查历史；仅处理未出现过的 move
        for last_move in last_moves:
            move_id = json.dumps(last_move, sort_keys=True)
            if move_id in self._seen_move_ids:
                continue
            self._seen_move_ids.add(move_id)
            new_move_count += 1

        self.logger.log_debug("EVENT_DETECT", f"New moves processed this step: {new_move_count}")

        # 无需再追踪长度，集合已记录
        
        # Get important info
        current_player = curr_obs['current_player']
        num_players = curr_obs['num_players']
        
        # Process ONLY new moves
        for last_move in last_moves:
            move_data = last_move['move']
            action_type = move_data.get('action_type')
            
            # Skip non-action moves (like DEAL)
            if last_move['player'] < 0:
                continue
                
            # Convert observer-relative player index to absolute
            acting_player_offset = last_move['player']  # relative to observer (me)
            # Correct absolute index of acting player from observer perspective
            acting_player_absolute = (self.my_player_number + acting_player_offset) % num_players
            self.logger.log_debug(
                "EVENT_DETECT", 
                f"Compute acting player: my_player={self.my_player_number}, offset={acting_player_offset} => abs={acting_player_absolute} (original curr_player field={current_player})"
            )
            
            # Check for different action types
            if action_type == 'REVEAL_COLOR':
                target_offset = move_data['target_offset']
                color = move_data['color']
                
                # Calculate absolute target player
                target_player_absolute = (acting_player_absolute + target_offset) % num_players
                target_offset_from_observer = (target_player_absolute - self.my_player_number) % num_players
                self.logger.log_debug(
                    "EVENT_DETECT", 
                    f"Compute target player: acting_abs={acting_player_absolute}, target_offset={target_offset} => abs={target_player_absolute}"
                )
                
                # Find which cards match (need observer-relative offset for observed_hands)
                
                events.append({
                    'clue_giver': f"P{acting_player_absolute + 1}",
                    'clue_recipient': f"P{target_player_absolute + 1}",
                    'clue_type': 'color',
                    'value': color,
                    'card_indices': self._find_matching_cards(curr_obs, target_offset_from_observer, 'color', color)
                })
                
            elif action_type == 'REVEAL_RANK':
                target_offset = move_data['target_offset']
                rank = move_data['rank']
                
                # Calculate absolute target player
                target_player_absolute = (acting_player_absolute + target_offset) % num_players
                target_offset_from_observer = (target_player_absolute - self.my_player_number) % num_players
                self.logger.log_debug(
                    "EVENT_DETECT", 
                    f"Compute target player: acting_abs={acting_player_absolute}, target_offset={target_offset} => abs={target_player_absolute}"
                )
                
                # Find which cards match (need observer-relative offset for observed_hands)
                
                events.append({
                    'clue_giver': f"P{acting_player_absolute + 1}",
                    'clue_recipient': f"P{target_player_absolute + 1}",
                    'clue_type': 'rank',
                    'value': rank,  # rank is already 1-indexed from LLM/game, keep it for belief graph
                    'card_indices': self._find_matching_cards(curr_obs, target_offset_from_observer, 'rank', rank)  # rank is 1-indexed
                })
                
            elif action_type == 'PLAY':
                events.append({
                    'type': 'play',
                    'player': f"P{acting_player_absolute + 1}",
                    'card_index': move_data.get('card_index', -1),
                    'success': True  # We can check fireworks to confirm
                })
                
            elif action_type == 'DISCARD':
                events.append({
                    'type': 'discard', 
                    'player': f"P{acting_player_absolute + 1}",
                    'card_index': last_move['move'].get('card_index', -1)
                })
        
        return events
    
    
    def _find_matching_cards(self, observation: Dict[str, Any], target_offset: int, clue_type: str, value) -> list:
        """Find which card indices match the given clue.
        
        For clues to ourselves (offset=0), use card_knowledge since we can't see our own cards.
        For clues to others, use observed_hands since we can see their cards.
        """
        matching_indices = []
        
        # Debug logging
        self.logger.log_debug("CARD_MATCHING_DEBUG", f"Looking for {clue_type}={value}, target_offset={target_offset}")
        
        if target_offset == 0:
            # Clue is for us - use card_knowledge
            if 'card_knowledge' in observation:
                my_knowledge = observation['card_knowledge'][0]  # Index 0 is always us
                for idx, card_knowledge in enumerate(my_knowledge):
                    if clue_type == 'color':
                        # Check if this card has this color
                        if card_knowledge.get('color') == value:
                            matching_indices.append(idx)
                    elif clue_type == 'rank':
                        # Check if this card has this rank (both are 0-indexed)
                        if card_knowledge.get('rank') == value:
                            matching_indices.append(idx)
            else:
                self.logger.log_debug("CARD_MATCHING_DEBUG", "No card_knowledge in observation")
        else:
            # Clue is for someone else - use observed_hands
            observed_hands = observation['observed_hands']
            if target_offset < len(observed_hands):
                hand = observed_hands[target_offset]
                self.logger.log_debug("CARD_MATCHING_DEBUG", f"Target hand: {hand}")
                
                for idx, card in enumerate(hand):
                    if clue_type == 'color':
                        card_color = card.get('color')
                        # Handle both string colors and numeric indices
                        if isinstance(card_color, str):
                            if card_color == value:
                                matching_indices.append(idx)
                        elif isinstance(card_color, int) and card_color >= 0:
                            card_color_char = ['R', 'Y', 'G', 'W', 'B'][card_color]
                            if card_color_char == value:
                                matching_indices.append(idx)
                    elif clue_type == 'rank':
                        card_rank = card.get('rank')
                        # value is 1-indexed (1-5) from LLM, card_rank is 0-indexed (0-4) from game
                        # Need to convert value to 0-indexed for comparison
                        if card_rank == value - 1:
                            matching_indices.append(idx)
            else:
                self.logger.log_debug("CARD_MATCHING_DEBUG", f"Target offset {target_offset} >= {len(observed_hands)}")
        
        self.logger.log_debug("CARD_MATCHING_DEBUG", f"Matching indices: {matching_indices}")
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
        self.belief_graph = {}
    
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
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with probabilistic belief graph."""
        # Determine our player number on first observation
        if self.my_player_number is None:
            self._determine_player_number(observation)
            self._initialize_belief_graph_with_player_number(observation)
            self._initialized = True
        
        # Always update beliefs when observing, regardless of whose turn it is
        if observation.get('last_moves'):
            self._update_beliefs_via_llm(observation)
        
        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None
        
        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
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
        # Get agent's actual player number (already determined)
        my_player_id = f"P{self.my_player_number + 1}"
        
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return
        
        prompt = f"""You are agent {my_player_id}. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

This event just occurred:
{json.dumps(event)}

Your task is to update your probabilistic model of beliefs.
For the hinted card, collapse its distribution to 100% for the hinted value.
For un-hinted cards, perform negative inference: set probability to 0 and re-normalize.

⚠️ INDEXING: card_indices uses 0-BASED ARRAY INDEXING ⚠️
   - Index 0 = P1_Card1 (first card)
   - Index 1 = P1_Card2 (second card)
   - Index 2 = P1_Card3 (third card)
   - Index 3 = P1_Card4 (fourth card)
   - Index 4 = P1_Card5 (fifth card)

Please reason step by step:
1. What does this event tell us?
2. Which cards were clued (map card_indices to P#_Card# keys using 0-based indexing)?
3. How should probability distributions be updated?
4. What negative inferences can we make for un-clued cards?

After your reasoning, provide the updated belief state in this JSON format:
{json.dumps(self._get_belief_template(), indent=2)}

CRITICAL RULES:
- MAPPING: card_indices are 0-indexed (index 0 → P1_Card1, index 1 → P1_Card2, etc.)
- CLUED cards (at indices in card_indices): Set probability to 1.0 for clued value, 0.0 for others
- UN-CLUED cards (indices 0-4 NOT in card_indices): Set probability to 0.0 for clued value, renormalize others
- Only update what is actually learned from THIS clue
- Apply negative inference correctly
- Preserve all fields including "actual_card_I_see" - only update belief distributions"""
        
        # Log the update prompt
        self.logger.log_info("BELIEF_UPDATE_PROMPT", f"Sending probabilistic update to LLM: {len(prompt)} chars")
        self.logger.log_debug("BELIEF_UPDATE_PROMPT_DETAIL", prompt)
        
        try:
            # First get the full reasoning response
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.7}
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
                            card_num = card_id.split('Card')[1]
                            
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
                            card_num = card_id.split('Card')[1]
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
        self.belief_graph = {}
    
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
        
        # Always update beliefs when observing, regardless of whose turn it is
        if observation.get('last_moves'):
            self._update_beliefs_with_tom(observation)
        
        # If not our turn, just record observation and return None
        if observation['current_player_offset'] != 0:
            self._add_observation_to_history(observation)
            return None
        
        # Our turn - augment observation with belief graph
        augmented_observation = observation.copy()
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
        # Get agent's actual player number (already determined)
        my_player_id = f"P{self.my_player_number + 1}"
        
        # Only update beliefs for clue events
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return
        
        prompt = f"""You are agent {my_player_id}. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event: {json.dumps(event)}

Your task is to perform Theory of Mind reasoning.
Analyze this action in the full context of your belief graph.
Does this action provide new evidence about the player's inferred_skill or signal a change in the team's Team_Focus?

Please reason step by step:
1. What does this event tell us about the player's skill level?
2. What might this reveal about team strategy?
3. How should both belief distributions AND ToM_Layer be updated?

Key principle: Use negative inference CORRECTLY

⚠️ INDEXING: card_indices uses 0-BASED ARRAY INDEXING ⚠️
   - Index 0 = P1_Card1 (first card)
   - Index 1 = P1_Card2 (second card)
   - Index 2 = P1_Card3 (third card)
   - Index 3 = P1_Card4 (fourth card)
   - Index 4 = P1_Card5 (fifth card)

- Cards at the indices in 'card_indices' HAVE the clued property (update to only that value)
- Cards NOT in 'card_indices' (other indices 0-4) do NOT have the clued property (remove ONLY the clued value, keep all other possibilities)
- Example: If clue is "rank 1" targeting card_indices [1,4] (0-indexed):
  * P1_Card1 (index 0): remove rank 1, keep ranks [2,3,4,5]
  * P1_Card2 (index 1): set to rank 1 ONLY ← clued
  * P1_Card3 (index 2): remove rank 1, keep ranks [2,3,4,5]
  * P1_Card4 (index 3): remove rank 1, keep ranks [2,3,4,5]
  * P1_Card5 (index 4): set to rank 1 ONLY ← clued
- Preserve the "actual_card_I_see" field - only update belief fields

Return the complete updated JSON graph."""
        
        # Log the ToM update prompt
        self.logger.log_info("TOM_UPDATE_PROMPT", f"Sending ToM update to LLM: {len(prompt)} chars")
        self.logger.log_debug("TOM_UPDATE_PROMPT_DETAIL", prompt)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'response_mime_type': 'application/json', 'temperature': 0.7}
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
                            card_num = card_id.split('Card')[1]
                            
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
                            card_num = card_id.split('Card')[1]
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
