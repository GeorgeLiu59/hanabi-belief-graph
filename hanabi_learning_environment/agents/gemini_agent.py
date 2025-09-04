# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gemini LLM Agent for Hanabi."""

import os
import json
import random
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from hanabi_learning_environment.rl_env import Agent

# Load environment variables
load_dotenv()

class GeminiAgent(Agent):
    """Agent that uses Gemini 2.0 Flash Lite to play Hanabi without any help."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the Gemini agent.
        
        Args:
            config: dict, With parameters for the game.
        """
        self.config = config
        
        # Try to load .env file explicitly
        load_dotenv()
        
        # Check for API key in multiple ways
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            # Try to read from .env file directly
            try:
                env_path = os.path.join(os.getcwd(), '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.startswith('GEMINI_API_KEY='):
                                self.api_key = line.split('=', 1)[1].strip().strip('"\'')
                                break
            except Exception as e:
                print(f"Warning: Could not read .env file: {e}")
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables or .env file. "
                "Please set GEMINI_API_KEY environment variable or create a .env file "
                "in the project root with: GEMINI_API_KEY=your_key_here"
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Game state tracking
        self.game_history = []
        self.current_game_id = None
        
    def reset(self, config):
        """Reset the agent for a new game."""
        self.config = config
        self.game_history = []
        self.current_game_id = random.randint(1000, 9999)
        
    def _format_observation_for_llm(self, observation: Dict[str, Any]) -> str:
        """Format the observation into a natural language description for the LLM."""
        if observation['current_player_offset'] != 0:
            return "It's not your turn."
            
        # Basic game state
        game_state = f"""
## GAME OVERVIEW
- Information tokens: {observation['information_tokens']}/8
- Life tokens: {observation['life_tokens']}/3
- Deck size: {observation['deck_size']}
- Number of players: {observation['num_players']}

## FIREWORKS PROGRESS (cards played so far)
"""
        
        for color, rank in observation['fireworks'].items():
            game_state += f"- {color}: {rank}/5 (need {rank+1} next)\n"
            
        # Discard pile analysis
        if observation['discard_pile']:
            game_state += f"\n## DISCARD PILE (what's been lost)\n"
            for card in observation['discard_pile']:
                game_state += f"- {card['color']}{card['rank']}\n"
        else:
            game_state += "\n## DISCARD PILE: empty (good - no cards lost yet)\n"
            
        # Other players' hands with strategic analysis
        game_state += "\n## OTHER PLAYERS' HANDS (what you can see)\n"
        for i, hand in enumerate(observation['observed_hands'][1:], 1):
            game_state += f"Player {i}: "
            for j, card in enumerate(hand):
                if card['color'] is not None and card['rank'] is not None:
                    game_state += f"{card['color']}{card['rank']} "
                else:
                    game_state += "? "
            game_state += "\n"
            
        # Your hand with knowledge analysis
        game_state += "\n## YOUR HAND (what you know vs. what you need)\n"
        my_hand = observation['observed_hands'][0]
        my_knowledge = observation['card_knowledge'][0]
        
        for i, (card, knowledge) in enumerate(zip(my_hand, my_knowledge)):
            game_state += f"Card {i}: "
            if knowledge['color'] is not None:
                game_state += f"Color: {knowledge['color']}, "
            else:
                game_state += "Color: unknown, "
                
            if knowledge['rank'] is not None:
                game_state += f"Rank: {knowledge['rank']}"
            else:
                game_state += "Rank: unknown"
            
            # Add strategic analysis
            if knowledge['color'] is not None and knowledge['rank'] is not None:
                current_firework = observation['fireworks'][knowledge['color']]
                if knowledge['rank'] == current_firework:
                    game_state += " ✅ PLAYABLE NOW!"
                elif knowledge['rank'] < current_firework:
                    game_state += " ❌ Too low - can't play yet"
                else:
                    game_state += f" ⏳ Need {current_firework} first"
            
            game_state += "\n"
            
        # Legal moves with strategic context
        game_state += "\n## AVAILABLE ACTIONS\n"
        for i, move in enumerate(observation['legal_moves']):
            if move['action_type'] == 'PLAY':
                game_state += f"{i}: Play card {move['card_index']}\n"
            elif move['action_type'] == 'DISCARD':
                game_state += f"{i}: Discard card {move['card_index']} (+1 info token)\n"
            elif move['action_type'] == 'REVEAL_COLOR':
                game_state += f"{i}: Hint color {move['color']} to player {move['target_offset']} (-1 info token)\n"
            elif move['action_type'] == 'REVEAL_RANK':
                game_state += f"{i}: Hint rank {move['rank']} to player {move['target_offset']} (-1 info token)\n"
                
        # Strategic recommendations
        game_state += "\n## STRATEGIC CONSIDERATIONS\n"
        if observation['information_tokens'] == 0:
            game_state += "- ⚠️ No info tokens left - must discard to get more\n"
        elif observation['information_tokens'] == 8:
            game_state += "- ⚠️ Max info tokens - should give hints or play cards\n"
        
        # Check for obvious plays
        obvious_plays = []
        for i, (card, knowledge) in enumerate(zip(my_hand, my_knowledge)):
            if knowledge['color'] is not None and knowledge['rank'] is not None:
                current_firework = observation['fireworks'][knowledge['color']]
                if knowledge['rank'] == current_firework:
                    obvious_plays.append(f"Card {i} ({knowledge['color']}{knowledge['rank']})")
        
        if obvious_plays:
            game_state += f"- 🎯 Obvious plays available: {', '.join(obvious_plays)}\n"
        else:
            game_state += "- 🤔 No obvious plays - consider hinting or discarding\n"
                
        return game_state
        
    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Create the prompt for the LLM."""
        game_state = self._format_observation_for_llm(observation)
        
        prompt = f"""You are playing Hanabi, a cooperative card game where you must work with your teammates to build fireworks (card sequences) in ascending order (0, 1, 2, 3, 4) for each color.

## GAME RULES
- Goal: Play cards in ascending order (0→1→2→3→4) for each color
- You cannot see your own cards, but can see other players' cards
- You receive hints about your cards from teammates
- You have 8 information tokens (for giving hints) and 3 life tokens
- Playing a wrong card costs 1 life token
- Game ends when you run out of life tokens or complete all fireworks

## STRATEGIC GUIDELINES
1. **RISK ASSESSMENT**: Only play cards you're confident about
2. **HINT STRATEGY**: Give hints that enable immediate plays or prevent discarding valuable cards
3. **RESOURCE MANAGEMENT**: Balance information tokens with strategic needs
4. **TEAM COORDINATION**: Work together to build sequences efficiently

## ACTION PRIORITY (in order):
1. **PLAY** a card if you're confident it's the next needed card
2. **HINT** if you can help a teammate play a card or save a valuable card
3. **DISCARD** only when you need information tokens and have no better options

## HINT STRATEGIES:
- **Playable hints**: Tell someone about a card they can play immediately
- **Save hints**: Prevent discarding cards needed for sequences
- **Finesse hints**: Enable plays through indirect information

## FORMAT REQUIREMENTS:
Respond with ONLY a valid JSON object in this exact format:
{{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 0-4, "target_offset": 1-3}}

**IMPORTANT**: 
- For PLAY/DISCARD: use "card_index" (0-4), set color/rank/target_offset to null
- For REVEAL_COLOR: use "color" and "target_offset", set card_index/rank to null  
- For REVEAL_RANK: use "rank" and "target_offset", set card_index/color to null

## CURRENT GAME STATE:
{game_state}

## YOUR TASK:
Analyze the game state and choose the most strategic action. Consider:
- What cards can be played safely?
- What hints would help your teammates most?
- How to manage information tokens efficiently?
- What risks are worth taking?

Choose the action that maximizes your team's chance of winning. Be strategic but don't be overly cautious - sometimes calculated risks are necessary!"""
        
        return prompt
        
    def _parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response and extract the action."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                action = json.loads(json_str)
                
                # Clean up the action to fix common LLM format issues
                action = self._clean_action_format(action)
                
                return action
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            pass
        return None
    
    def _clean_action_format(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up common LLM format issues in actions."""
        cleaned = action.copy()
        
        # Fix card_index issues
        if 'card_index' in cleaned:
            if cleaned['card_index'] == -1 or cleaned['card_index'] is None:
                # For hints, card_index should be None
                if cleaned['action_type'] in ['REVEAL_COLOR', 'REVEAL_RANK']:
                    cleaned['card_index'] = None
                # For play/discard, card_index should be 0-4
                elif cleaned['action_type'] in ['PLAY', 'DISCARD']:
                    cleaned['card_index'] = 0  # Default to first card
        
        # Ensure required fields are present and correct
        if cleaned['action_type'] == 'PLAY':
            if 'card_index' not in cleaned or cleaned['card_index'] is None:
                cleaned['card_index'] = 0
            cleaned['color'] = None
            cleaned['rank'] = None
            cleaned['target_offset'] = None
            
        elif cleaned['action_type'] == 'DISCARD':
            if 'card_index' not in cleaned or cleaned['card_index'] is None:
                cleaned['card_index'] = 0
            cleaned['color'] = None
            cleaned['rank'] = None
            cleaned['target_offset'] = None
            
        elif cleaned['action_type'] == 'REVEAL_COLOR':
            if 'color' not in cleaned or cleaned['color'] is None:
                cleaned['color'] = 'R'  # Default color
            if 'target_offset' not in cleaned or cleaned['target_offset'] is None:
                cleaned['target_offset'] = 1  # Default target
            cleaned['card_index'] = None
            cleaned['rank'] = None
            
        elif cleaned['action_type'] == 'REVEAL_RANK':
            if 'rank' not in cleaned or cleaned['rank'] is None:
                cleaned['rank'] = 0  # Default rank
            if 'target_offset' not in cleaned or cleaned['target_offset'] is None:
                cleaned['target_offset'] = 1  # Default target
            cleaned['card_index'] = None
            cleaned['color'] = None
        
        return cleaned
        
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act based on an observation using Gemini LLM."""
        if observation['current_player_offset'] != 0:
            return None
            
        try:
            # Create prompt
            prompt = self._create_prompt(observation)
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse response
            action = self._parse_llm_response(response_text)
            
            if action is None:
                # Fallback to random action if parsing fails
                print(f"Failed to parse LLM response: {response_text}")
                return random.choice(observation['legal_moves'])
                
            # Validate action is legal
            legal_moves = observation['legal_moves']
            for legal_move in legal_moves:
                if (action.get('action_type') == legal_move.get('action_type') and
                    action.get('card_index') == legal_move.get('card_index') and
                    action.get('color') == legal_move.get('color') and
                    action.get('rank') == legal_move.get('rank') and
                    action.get('target_offset') == legal_move.get('target_offset')):
                    return action
                    
            # If action is not legal, choose random legal action
            print(f"LLM action not legal: {action}")
            return random.choice(observation['legal_moves'])
            
        except Exception as e:
            print(f"Error in Gemini agent: {e}")
            # Fallback to random action
            return random.choice(observation['legal_moves'])
