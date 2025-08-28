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
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
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
Game State:
- Information tokens: {observation['information_tokens']}
- Life tokens: {observation['life_tokens']}
- Deck size: {observation['deck_size']}
- Number of players: {observation['num_players']}

Fireworks (cards played so far):
"""
        
        for color, rank in observation['fireworks'].items():
            game_state += f"- {color}: {rank}\n"
            
        # Discard pile
        if observation['discard_pile']:
            game_state += f"\nDiscard pile: {observation['discard_pile']}\n"
        else:
            game_state += "\nDiscard pile: empty\n"
            
        # Other players' hands (what you can see)
        game_state += "\nOther players' hands:\n"
        for i, hand in enumerate(observation['observed_hands'][1:], 1):
            game_state += f"Player {i}: "
            for j, card in enumerate(hand):
                if card['color'] is not None and card['rank'] is not None:
                    game_state += f"{card['color']}{card['rank']} "
                else:
                    game_state += "? "
            game_state += "\n"
            
        # Your hand (what you know about it)
        game_state += "\nYour hand:\n"
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
            game_state += "\n"
            
        # Legal moves
        game_state += "\nLegal moves:\n"
        for i, move in enumerate(observation['legal_moves']):
            if move['action_type'] == 'PLAY':
                game_state += f"{i}: Play card {move['card_index']}\n"
            elif move['action_type'] == 'DISCARD':
                game_state += f"{i}: Discard card {move['card_index']}\n"
            elif move['action_type'] == 'REVEAL_COLOR':
                game_state += f"{i}: Hint color {move['color']} to player {move['target_offset']}\n"
            elif move['action_type'] == 'REVEAL_RANK':
                game_state += f"{i}: Hint rank {move['rank']} to player {move['target_offset']}\n"
                
        return game_state
        
    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Create the prompt for the LLM."""
        game_state = self._format_observation_for_llm(observation)
        
        prompt = f"""You are playing Hanabi, a cooperative card game. You cannot see your own cards, but you can see other players' cards and receive hints about your cards.

Game Rules:
- The goal is to play cards in ascending order (0, 1, 2, 3, 4) for each color
- You can play a card if it's the next card needed for that color's sequence
- You can discard a card to gain an information token
- You can give hints to other players about colors or ranks of their cards
- You have limited information tokens (8 max) and life tokens (3 max)
- If you play a wrong card, you lose a life token
- The game ends when you run out of life tokens or successfully play all cards

{game_state}

Based on the current game state, choose the best action. Respond with ONLY a JSON object in this exact format:
{{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 0-4, "target_offset": 1-3}}

Choose the action that you think will help the team win. Be strategic about when to play, discard, or give hints."""
        
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
                return action
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None
        
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
