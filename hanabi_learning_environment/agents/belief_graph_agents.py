"""Belief graph scaffolding agents - EXACTLY as specified.

These are thin layers on GeminiAgent that maintain JSON belief structures
and pass them to the LLM for reasoning. No complex graph infrastructure needed.
"""

import json
from typing import Dict, Any, Optional

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
        self.previous_observation = None
    
    def reset(self, config):
        """Reset and initialize belief graph."""
        super().reset(config)
        
        # Initialize belief graph EXACTLY as specified
        self.belief_graph = {
            "GameState": {
                "clues": 8,
                "life": 3,
                "deck_size": 50
            },
            "My_Hand_Beliefs": {},
            "Teammate_Hand_Beliefs": {}
        }
        
        # Initialize my hand beliefs (I can't see my cards)
        for i in range(5):  # Assuming 5 cards per hand
            self.belief_graph["My_Hand_Beliefs"][f"P1_Card{i+1}"] = {
                "possible_colors": ["red", "blue", "green", "white", "yellow"],
                "possible_ranks": [1, 2, 3, 4, 5]
            }
        
        # Initialize teammate beliefs (I can see their cards but they can't)
        num_players = config.get('players', 2)
        for player in range(2, num_players + 1):
            self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"] = {}
            for card in range(5):
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"][f"P{player}_Card{card+1}"] = {
                    "actual_card_I_see": "Unknown",  # Will be updated from observation
                    f"p{player}_belief": {
                        "possible_colors": ["red", "blue", "green", "white", "yellow"],
                        "possible_ranks": [1, 2, 3, 4, 5]
                    }
                }
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with belief graph augmentation."""
        # Always check for events and update beliefs, even when not our turn
        if self.previous_observation:
            # Check if any events happened since last observation
            self._update_beliefs_via_llm(observation)
        else:
            # First observation - initialize what I can see
            self._initialize_visible_cards(observation)
        
        # Store observation for next time
        self.previous_observation = observation.copy()
        
        # If not our turn, just observe
        if observation['current_player_offset'] != 0:
            return None
        
        # Our turn - add belief graph to observation for GeminiAgent
        augmented_observation = observation.copy()
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'certainty'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()
        
        # Log current belief state
        self.logger.log_info("BELIEF_STATE", f"Certainty graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))
        
        # Let GeminiAgent handle the decision with our belief graph
        action = super().act(augmented_observation)
        
        return action
    
    def _format_belief_graph_natural_language(self) -> str:
        """Convert belief graph to natural language for easier LLM understanding."""
        nl_description = "## BELIEF GRAPH ANALYSIS (CERTAINTY VARIANT)\n\n"
        
        # Game state
        gs = self.belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"
        
        # My hand beliefs
        nl_description += "**MY HAND BELIEFS:**\n"
        for card_id, beliefs in self.belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            colors = beliefs['possible_colors']
            ranks = beliefs['possible_ranks']
            
            if len(colors) == 1 and len(ranks) == 1:
                nl_description += f"- Card {card_num}: KNOWN to be {colors[0].upper()} {ranks[0]}\n"
            elif len(colors) == 1:
                nl_description += f"- Card {card_num}: KNOWN color {colors[0].upper()}, could be rank {ranks}\n"
            elif len(ranks) == 1:
                nl_description += f"- Card {card_num}: KNOWN rank {ranks[0]}, could be {colors}\n"
            else:
                nl_description += f"- Card {card_num}: UNKNOWN - could be {len(colors)} colors × {len(ranks)} ranks\n"
        
        # Teammate beliefs
        nl_description += "\n**TEAMMATE KNOWLEDGE MODEL:**\n"
        for player_hand, hand_data in self.belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            nl_description += f"\nPlayer {player_num}'s cards (what I see vs what they know):\n"
            
            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data['actual_card_I_see']
                belief = card_data[f'p{player_num}_belief']
                colors = belief['possible_colors']
                ranks = belief['possible_ranks']
                
                if len(colors) == 1 and len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they KNOW it's {colors[0].upper()} {ranks[0]}\n"
                elif len(colors) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know COLOR {colors[0].upper()} only\n"
                elif len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know RANK {ranks[0]} only\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know NOTHING\n"
        
        return nl_description
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Update belief graph with visible cards."""
        # Update teammate cards I can see
        for player_idx, hand in enumerate(observation.get('observed_hands', [])):
            if player_idx == 0:
                continue  # Skip my own hand
            
            player_id = f"P{player_idx + 1}"
            if f"{player_id}_Hand" in self.belief_graph["Teammate_Hand_Beliefs"]:
                for card_idx, card in enumerate(hand):
                    if card.get('color') and card.get('rank') is not None:
                        card_key = f"{player_id}_Card{card_idx + 1}"
                        color = card['color']
                        rank = card['rank'] + 1 if card['rank'] >= 0 else '?'
                        self.belief_graph["Teammate_Hand_Beliefs"][f"{player_id}_Hand"][card_key]["actual_card_I_see"] = f"{color} {rank}"
    
    def _update_beliefs_via_llm(self, observation: Dict[str, Any]):
        """Update beliefs using LLM as specified."""
        # Detect what happened
        event = self._detect_event(self.previous_observation, observation)
        if not event:
            self.logger.log_debug("EVENT_DETECTION", "No event detected")
            return
        
        self.logger.log_info("EVENT_DETECTED", f"Event: {json.dumps(event)}")
        
        # Check if this is about our own hand or teammate's
        if event.get('clue_recipient') == 'P1':
            # Update our own beliefs
            prompt = f"""You are agent P1. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just received a clue:
{json.dumps(event, indent=2)}

Your task is to update your OWN beliefs (My_Hand_Beliefs) based on the clue received.
For the Certainty Graph, narrow down possible values for the hinted cards.
Apply negative inference for un-hinted cards.

Before returning the updated graph, explain your reasoning:
1. Which of my cards match the clue?
2. How should I narrow down the possibilities for those cards?
3. What negative inferences can I make about my other cards?

Return the complete, updated JSON graph."""
        else:
            # Update teammate beliefs
            prompt = f"""You are agent P1. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event:
{json.dumps(event, indent=2)}

Your task is to update your model of your teammate's beliefs. 
For the Certainty Graph, narrow down possible values based on the clue.
Apply negative inference where appropriate.

Before returning the updated graph, explain your reasoning:
1. What does this event tell us?
2. How should beliefs be updated?
3. What negative inferences can we make?

Return the complete, updated JSON graph."""
        
        # Log the update prompt
        self.logger.log_info("BELIEF_UPDATE_PROMPT", f"Sending belief update to LLM: {len(prompt)} chars")
        self.logger.log_debug("BELIEF_UPDATE_PROMPT_DETAIL", prompt)
        
        try:
            # Get LLM to update the graph
            response = self.model.generate_content(
                prompt,
                generation_config={'response_mime_type': 'application/json', 'temperature': 0.7}
            )
            
            # Log the LLM response
            self.logger.log_info("BELIEF_UPDATE_RESPONSE", f"LLM response: {len(response.text)} chars")
            self.logger.log_debug("BELIEF_UPDATE_RESPONSE_DETAIL", response.text)
            
            # Update our belief graph
            self.belief_graph = json.loads(response.text)
            self.logger.log_info("LLM_UPDATE", "Belief graph updated via LLM")
            
        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update beliefs: {e}")
    
    def _detect_event(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]) -> Optional[Dict]:
        """Detect game events by comparing observations."""
        # Log what we're looking for
        self.logger.log_debug("EVENT_DETECTION_CHECK", f"Checking for events. Keys in observation: {list(curr_obs.keys())}")
        
        # Get the last move from history if available
        last_moves = curr_obs.get('last_moves', [])
        if not last_moves:
            self.logger.log_debug("EVENT_DETECTION", "No last_moves in observation")
            return None
            
        # Look at the most recent move
        last_move = last_moves[-1]
        move_type = last_move.get('move', {}).get('type')
        
        self.logger.log_debug("EVENT_DETECTION_MOVE", f"Last move: {last_move}, move type: {move_type}")
        
        # Map move types: 0=PLAY, 1=DISCARD, 2=REVEAL_COLOR, 3=REVEAL_RANK
        if move_type == 2:  # REVEAL_COLOR
            move = last_move['move']
            # Get affected cards from the move
            target_offset = move.get('target', -1)
            color = move.get('color', 0)
            color_names = ['R', 'Y', 'G', 'W', 'B']
            
            return {
                'clue_giver': f"P{last_move.get('player', 0) + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'color',
                'value': color_names[color] if color < len(color_names) else 'unknown',
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'color', color)
            }
            
        elif move_type == 3:  # REVEAL_RANK  
            move = last_move['move']
            target_offset = move.get('target', -1)
            rank = move.get('rank', -1)
            
            return {
                'clue_giver': f"P{last_move.get('player', 0) + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'rank',
                'value': rank + 1 if rank >= 0 else 'unknown',
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'rank', rank)
            }
            
        elif move_type == 0:  # PLAY
            return {
                'type': 'play',
                'player': f"P{last_move.get('player', 0) + 1}",
                'card_index': last_move['move'].get('card_index', -1),
                'success': True  # We can check fireworks to confirm
            }
            
        elif move_type == 1:  # DISCARD
            return {
                'type': 'discard', 
                'player': f"P{last_move.get('player', 0) + 1}",
                'card_index': last_move['move'].get('card_index', -1)
            }
        
        return None
    
    def _find_matching_cards(self, observation: Dict[str, Any], target_offset: int, clue_type: str, value: int) -> list:
        """Find which card indices match the given clue."""
        matching_indices = []
        
        # Get the target player's hand
        if target_offset < len(observation.get('observed_hands', [])):
            hand = observation['observed_hands'][target_offset]
            for idx, card in enumerate(hand):
                if clue_type == 'color' and card.get('color', -1) == value:
                    matching_indices.append(idx)
                elif clue_type == 'rank' and card.get('rank', -1) == value:
                    matching_indices.append(idx)
        
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
        self.previous_observation = None
    
    def reset(self, config):
        """Reset and initialize probabilistic belief graph."""
        super().reset(config)
        
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
        
        # My hand with probability distributions
        for i in range(5):
            self.belief_graph["My_Hand_Beliefs"][f"P1_Card{i+1}"] = {
                "color_distribution": uniform_color.copy(),
                "rank_distribution": uniform_rank.copy()
            }
        
        # Teammate hands with distributions
        num_players = config.get('players', 2)
        for player in range(2, num_players + 1):
            self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"] = {}
            for card in range(5):
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"][f"P{player}_Card{card+1}"] = {
                    "actual_card_I_see": "Unknown",
                    f"p{player}_belief": {
                        "color_distribution": uniform_color.copy(),
                        "rank_distribution": uniform_rank.copy()
                    }
                }
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with probabilistic belief graph."""
        # Always check for events and update beliefs
        if self.previous_observation:
            self._update_beliefs_via_llm(observation)
        else:
            self._initialize_visible_cards(observation)
        
        # Store observation for next time
        self.previous_observation = observation.copy()
        
        # If not our turn, just observe
        if observation['current_player_offset'] != 0:
            return None
        
        # Our turn - add belief graph to observation
        augmented_observation = observation.copy()
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'probabilistic'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()
        
        self.logger.log_info("BELIEF_STATE", f"Probabilistic graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))
        
        action = super().act(augmented_observation)
        
        return action
    
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
                actual = card_data['actual_card_I_see']
                belief = card_data[f'p{player_num}_belief']
                color_dist = belief['color_distribution']
                rank_dist = belief['rank_distribution']
                
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
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Update with visible cards."""
        for player_idx, hand in enumerate(observation.get('observed_hands', [])):
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
    
    def _update_beliefs_via_llm(self, observation: Dict[str, Any]):
        """Update probabilistic beliefs via LLM."""
        event = self._detect_event(self.previous_observation, observation)
        if not event:
            return
        
        # Create prompt as specified
        prompt = f"""You are agent P1. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event:
{json.dumps(event, indent=2)}

Your task is to update your probabilistic model of your teammate's beliefs.
For the hinted card, collapse its distribution to 100% for the hinted value.
For un-hinted cards, perform negative inference: set probability to 0 and re-normalize.

Before returning the updated graph, explain your reasoning:
1. What does this event tell us?
2. How should probability distributions be updated?
3. What negative inferences can we make?

Return the complete, updated JSON graph."""
        
        # Log the update prompt
        self.logger.log_info("BELIEF_UPDATE_PROMPT", f"Sending probabilistic update to LLM: {len(prompt)} chars")
        self.logger.log_debug("BELIEF_UPDATE_PROMPT_DETAIL", prompt)
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'response_mime_type': 'application/json', 'temperature': 0.7}
            )
            
            # Log response
            self.logger.log_info("BELIEF_UPDATE_RESPONSE", f"LLM response: {len(response.text)} chars")
            self.logger.log_debug("BELIEF_UPDATE_RESPONSE_DETAIL", response.text)
            
            self.belief_graph = json.loads(response.text)
            self.logger.log_info("LLM_UPDATE", "Probabilistic belief graph updated via LLM")
            
        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update probabilistic beliefs: {e}")
    
    def _detect_event(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]) -> Optional[Dict]:
        """Detect game events by comparing observations."""
        # Log what we're looking for
        self.logger.log_debug("EVENT_DETECTION_CHECK", f"Checking for events. Keys in observation: {list(curr_obs.keys())}")
        
        # Get the last move from history if available
        last_moves = curr_obs.get('last_moves', [])
        if not last_moves:
            self.logger.log_debug("EVENT_DETECTION", "No last_moves in observation")
            return None
            
        # Look at the most recent move
        last_move = last_moves[-1]
        move_type = last_move.get('move', {}).get('type')
        
        self.logger.log_debug("EVENT_DETECTION_MOVE", f"Last move: {last_move}, move type: {move_type}")
        
        # Map move types: 0=PLAY, 1=DISCARD, 2=REVEAL_COLOR, 3=REVEAL_RANK
        if move_type == 2:  # REVEAL_COLOR
            move = last_move['move']
            # Get affected cards from the move
            target_offset = move.get('target', -1)
            color = move.get('color', 0)
            color_names = ['R', 'Y', 'G', 'W', 'B']
            
            return {
                'clue_giver': f"P{last_move.get('player', 0) + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'color',
                'value': color_names[color] if color < len(color_names) else 'unknown',
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'color', color)
            }
            
        elif move_type == 3:  # REVEAL_RANK  
            move = last_move['move']
            target_offset = move.get('target', -1)
            rank = move.get('rank', -1)
            
            return {
                'clue_giver': f"P{last_move.get('player', 0) + 1}",
                'clue_recipient': f"P{target_offset + 1}",
                'clue_type': 'rank',
                'value': rank + 1 if rank >= 0 else 'unknown',
                'card_indices': self._find_matching_cards(curr_obs, target_offset, 'rank', rank)
            }
            
        elif move_type == 0:  # PLAY
            return {
                'type': 'play',
                'player': f"P{last_move.get('player', 0) + 1}",
                'card_index': last_move['move'].get('card_index', -1),
                'success': True  # We can check fireworks to confirm
            }
            
        elif move_type == 1:  # DISCARD
            return {
                'type': 'discard', 
                'player': f"P{last_move.get('player', 0) + 1}",
                'card_index': last_move['move'].get('card_index', -1)
            }
        
        return None
    
    def _find_matching_cards(self, observation: Dict[str, Any], target_offset: int, clue_type: str, value: int) -> list:
        """Find which card indices match the given clue."""
        matching_indices = []
        
        # Get the target player's hand
        if target_offset < len(observation.get('observed_hands', [])):
            hand = observation['observed_hands'][target_offset]
            for idx, card in enumerate(hand):
                if clue_type == 'color' and card.get('color', -1) == value:
                    matching_indices.append(idx)
                elif clue_type == 'rank' and card.get('rank', -1) == value:
                    matching_indices.append(idx)
        
        return matching_indices


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
        self.previous_observation = None
    
    def reset(self, config):
        """Reset with ToM layer."""
        super().reset(config)
        
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
        
        # Probabilistic beliefs for cards
        for i in range(5):
            self.belief_graph["My_Hand_Beliefs"][f"P1_Card{i+1}"] = {
                "color_distribution": uniform_color.copy(),
                "rank_distribution": uniform_rank.copy()
            }
        
        # Teammate beliefs with ToM models
        num_players = config.get('players', 2)
        for player in range(2, num_players + 1):
            self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"] = {}
            for card in range(5):
                self.belief_graph["Teammate_Hand_Beliefs"][f"P{player}_Hand"][f"P{player}_Card{card+1}"] = {
                    "actual_card_I_see": "Unknown",
                    f"p{player}_belief": {
                        "color_distribution": uniform_color.copy(),
                        "rank_distribution": uniform_rank.copy()
                    }
                }
            
            # ToM model for each teammate
            self.belief_graph["ToM_Layer"]["Teammates"][f"P{player}"] = {
                "inferred_skill": 0.5,
                "play_aggressiveness": 0.5,
                "hint_quality": 0.5
            }
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act with ToM-enhanced belief graph."""
        # Always check for events and update beliefs
        if self.previous_observation:
            self._update_beliefs_with_tom(observation)
        else:
            self._initialize_visible_cards(observation)
        
        # Store observation for next time
        self.previous_observation = observation.copy()
        
        # If not our turn, just observe
        if observation['current_player_offset'] != 0:
            return None
        
        # Our turn - add to observation
        augmented_observation = observation.copy()
        augmented_observation['belief_graph'] = self.belief_graph
        augmented_observation['belief_variant'] = 'theory_of_mind'
        augmented_observation['belief_graph_natural_language'] = self._format_belief_graph_natural_language()
        
        self.logger.log_info("BELIEF_STATE", f"ToM graph size: {len(json.dumps(self.belief_graph))} chars")
        self.logger.log_debug("BELIEF_GRAPH_DETAIL", json.dumps(self.belief_graph, indent=2))
        
        action = super().act(augmented_observation)
        
        return action
    
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
            
            nl_description += f"- {player}: {skill_desc} player (skill: {skill:.1f}), {aggr_desc} style\n"
        
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
                actual = card_data['actual_card_I_see']
                nl_description += f"  - Card {card_num}: I see {actual}\n"
        
        return nl_description
    
    def _initialize_visible_cards(self, observation: Dict[str, Any]):
        """Initialize visible cards."""
        for player_idx, hand in enumerate(observation.get('observed_hands', [])):
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
        event = self._detect_event(self.previous_observation, observation)
        if not event:
            return
        
        # ToM prompt as specified
        prompt = f"""You are agent P1. Here is your current belief graph:
{json.dumps(self.belief_graph, indent=2)}

You just observed this event:
{json.dumps(event, indent=2)}

Your task is to perform Theory of Mind reasoning.
Analyze this action in the full context of your belief graph.
Does this action provide new evidence about the player's inferred_skill or signal a change in the team's Team_Focus?

Before returning the updated graph, explain your reasoning:
1. What does this event tell us about the player's skill level?
2. What might this reveal about team strategy?
3. How should both belief distributions AND ToM_Layer be updated?

Return the complete, updated JSON graph..."""
        
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
            
            self.belief_graph = json.loads(response.text)
            
            # Log ToM insights
            if 'ToM_Layer' in self.belief_graph:
                self.logger.log_info("TOM_UPDATE", f"Team focus: {self.belief_graph['ToM_Layer']['Team_Focus']}")
                self.logger.log_info("LLM_UPDATE", "ToM belief graph updated via LLM")
            
        except Exception as e:
            self.logger.log_error("TOM_ERROR", f"Failed to update ToM beliefs: {e}")
    
    def _detect_event(self, prev_obs: Dict[str, Any], curr_obs: Dict[str, Any]) -> Optional[Dict]:
        """Detect game events by comparing observations."""
        # Check for clues given
        if prev_obs.get('information_tokens', 0) > curr_obs.get('information_tokens', 0):
            # A clue was given - check last moves to find details
            last_moves = curr_obs.get('last_moves', [])
            if last_moves:
                last_move = last_moves[-1]
                if last_move.get('move', {}).get('type') in [2, 3]:  # REVEAL_COLOR or REVEAL_RANK
                    move = last_move['move']
                    return {
                        'action': 'CLUE',
                        'clue_giver': f"P{last_move.get('player', 0) + 1}",
                        'clue_recipient': f"P{move.get('target', 0) + 1}",
                        'clue_type': 'color' if move['type'] == 2 else 'rank',
                        'value': move.get('color', move.get('rank', 'unknown')),
                        'card_indices': move.get('card_indices', [])
                    }
        
        # Check for plays
        curr_fireworks = sum(curr_obs.get('fireworks', {}).values())
        prev_fireworks = sum(prev_obs.get('fireworks', {}).values())
        if curr_fireworks > prev_fireworks:
            # A card was successfully played
            last_moves = curr_obs.get('last_moves', [])
            if last_moves:
                last_move = last_moves[-1]
                if last_move.get('move', {}).get('type') == 0:  # PLAY
                    return {
                        'action': 'PLAY',
                        'player': f"P{last_move.get('player', 0) + 1}",
                        'card_index': last_move['move'].get('card_index', -1),
                        'success': True
                    }
        
        # Check for discards
        if curr_obs.get('deck_size', 0) < prev_obs.get('deck_size', 0):
            last_moves = curr_obs.get('last_moves', [])
            if last_moves:
                last_move = last_moves[-1]
                if last_move.get('move', {}).get('type') == 1:  # DISCARD
                    return {
                        'action': 'DISCARD',
                        'player': f"P{last_move.get('player', 0) + 1}",
                        'card_index': last_move['move'].get('card_index', -1)
                    }
        
        return None
