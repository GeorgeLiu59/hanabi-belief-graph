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
import time
import numpy as np
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
        
        # Game state tracking - similar to Rainbow agent's observation stacking
        self.game_history = []
        self.current_game_id = None
        self.observation_history = []  # Store previous observations
        self.history_size = 4  # Match Rainbow agent's default history size
        self.action_history = []  # Store previous actions taken
        
        # Rate limiting tracking
        self.request_times = []  # Track timestamps of API requests
        self.max_requests_per_minute = 30  # Exact limit per minute
        self.current_minute = None  # Track which minute we're in
        
    def reset(self, config):
        """Reset the agent for a new game."""
        self.config = config
        self.game_history = []
        self.observation_history = []
        self.action_history = []
        self.current_game_id = random.randint(1000, 9999)
        
    def _add_observation_to_history(self, observation: Dict[str, Any]):
        """Add current observation to history, maintaining the history size limit."""
        # Store a summary of the observation for history
        obs_summary = {
            'turn': len(self.observation_history) + 1,
            'information_tokens': observation['information_tokens'],
            'life_tokens': observation['life_tokens'],
            'deck_size': observation['deck_size'],
            'fireworks': observation['fireworks'].copy(),
            'discard_pile_size': len(observation['discard_pile']),
            'discard_pile': observation['discard_pile'].copy(),
            'my_hand': observation['observed_hands'][0].copy(),
            'my_knowledge': observation['card_knowledge'][0].copy(),
            'other_hands': [hand.copy() for hand in observation['observed_hands'][1:]],
            'legal_moves_count': len(observation['legal_moves'])
        }
        
        self.observation_history.append(obs_summary)
        
        # Maintain history size limit
        if len(self.observation_history) > self.history_size:
            self.observation_history.pop(0)
    
    def _add_action_to_history(self, action: Dict[str, Any]):
        """Add the action taken to history."""
        self.action_history.append(action.copy())
        
        # Maintain history size limit
        if len(self.action_history) > self.history_size:
            self.action_history.pop(0)
    
    def _format_history_for_llm(self) -> str:
        """Format the observation and action history for the LLM."""
        if not self.observation_history:
            return "## GAME HISTORY: This is the first turn of the game.\n"
        
        history_text = "## GAME HISTORY (Previous Turns)\n"
        
        for i, (obs, action) in enumerate(zip(self.observation_history, self.action_history)):
            turn_num = obs['turn']
            history_text += f"\n**Turn {turn_num}:**\n"
            
            # Show what happened in this turn
            if action['action_type'] == 'PLAY':
                history_text += f"- You PLAYED card {action['card_index']}\n"
            elif action['action_type'] == 'DISCARD':
                history_text += f"- You DISCARDED card {action['card_index']}\n"
            elif action['action_type'] == 'REVEAL_COLOR':
                history_text += f"- You gave a COLOR hint ({action['color']}) to player {action['target_offset']}\n"
            elif action['action_type'] == 'REVEAL_RANK':
                history_text += f"- You gave a RANK hint ({action['rank']}) to player {action['target_offset']}\n"
            
            # Show state changes
            if i > 0:
                prev_obs = self.observation_history[i-1]
                if obs['information_tokens'] != prev_obs['information_tokens']:
                    change = obs['information_tokens'] - prev_obs['information_tokens']
                    if change > 0:
                        history_text += f"- Gained {change} information token(s)\n"
                    else:
                        history_text += f"- Used {abs(change)} information token(s)\n"
                
                if obs['life_tokens'] != prev_obs['life_tokens']:
                    change = obs['life_tokens'] - prev_obs['life_tokens']
                    if change < 0:
                        history_text += f"- Lost {abs(change)} life token(s) (mistake made)\n"
                
                if obs['deck_size'] != prev_obs['deck_size']:
                    history_text += f"- Deck size changed from {prev_obs['deck_size']} to {obs['deck_size']}\n"
                
                # Show fireworks progress
                for color in obs['fireworks']:
                    if obs['fireworks'][color] != prev_obs['fireworks'][color]:
                        history_text += f"- {color} firework progressed from {prev_obs['fireworks'][color]} to {obs['fireworks'][color]}\n"
                
                # Show discard pile changes
                if obs['discard_pile_size'] != prev_obs['discard_pile_size']:
                    new_cards = obs['discard_pile_size'] - prev_obs['discard_pile_size']
                    if new_cards > 0:
                        history_text += f"- {new_cards} card(s) added to discard pile\n"
        
        history_text += "\n**Key Insights from History:**\n"
        
        # Analyze patterns in the history
        if len(self.action_history) >= 2:
            recent_actions = [action['action_type'] for action in self.action_history[-3:]]
            if recent_actions.count('REVEAL_COLOR') + recent_actions.count('REVEAL_RANK') >= 2:
                history_text += "- You've been giving many hints recently - consider playing cards or discarding\n"
            elif recent_actions.count('PLAY') >= 2:
                history_text += "- You've been playing cards successfully - good progress!\n"
            elif recent_actions.count('DISCARD') >= 2:
                history_text += "- You've been discarding frequently - make sure you're not discarding important cards\n"
        
        # Check for information token management
        if self.observation_history:
            current_tokens = self.observation_history[-1]['information_tokens']
            if current_tokens == 0:
                history_text += "- ⚠️ No information tokens left - must discard to get more\n"
            elif current_tokens == 8:
                history_text += "- ⚠️ Maximum information tokens - should give hints or play cards\n"
        
        return history_text
    
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
    
    def _check_rate_limit(self):
        """Check if we're at the rate limit and wait for the next minute if necessary."""
        current_time = time.time()
        current_minute = int(current_time // 60)  # Which minute we're in
        
        # If we've moved to a new minute, reset the counter
        if self.current_minute != current_minute:
            self.request_times = []
            self.current_minute = current_minute
        
        # If we've hit the limit for this minute, wait until the next minute
        if len(self.request_times) >= self.max_requests_per_minute:
            # Calculate seconds until next minute
            seconds_until_next_minute = 60 - (current_time % 60) + 1  # Add 1 second buffer
            
            print(f"Rate limit reached ({len(self.request_times)}/{self.max_requests_per_minute} requests this minute). Waiting {seconds_until_next_minute:.1f} seconds until next minute...")
            time.sleep(seconds_until_next_minute)
            
            # Reset for the new minute
            self.request_times = []
            self.current_minute = int(time.time() // 60)
        
        # Record this request BEFORE making the API call
        self.request_times.append(current_time)
        
    def _create_prompt(self, observation: Dict[str, Any]) -> str:
        """Create the prompt for the LLM."""
        game_state = self._format_observation_for_llm(observation)
        history = self._format_history_for_llm()
        
        prompt = f"""You are playing Hanabi-Full-CardKnowledge, a cooperative card game where you must work with your teammates to build fireworks (card sequences) in ascending order (0, 1, 2, 3, 4) for each color.

{history}

## GAME RULES (CardKnowledge Variant)
- Goal: Play cards in ascending order (0→1→2→3→4) for each color
- **IMPORTANT**: You CAN see your own cards (this is the CardKnowledge variant)
- You can see other players' cards and they can see yours
- You have 8 information tokens (for giving hints) and 3 life tokens
- Playing a wrong card costs 1 life token
- Game ends when you run out of life tokens, deck is empty, or complete all fireworks
- Maximum score is 25 points (5 colors × 5 ranks)

## ADVANCED STRATEGIC GUIDELINES
1. **CONFIDENCE PLAYING**: Only play cards you're 100% certain about
2. **HINT EFFICIENCY**: Give hints that create immediate plays or save critical cards
3. **TOKEN MANAGEMENT**: Balance information tokens - don't waste them on obvious plays
4. **SEQUENCE BUILDING**: Focus on completing one color at a time when possible
5. **ENDGAME AWARENESS**: When deck is low, prioritize plays over hints

## ACTION PRIORITY (in order):
1. **PLAY** a card you're certain is the next needed card (highest priority)
2. **HINT** to enable a teammate's immediate play or save a critical card
3. **DISCARD** when you need info tokens and have no better options
4. **DISCARD** obviously useless cards (duplicates, too-low ranks)

## ADVANCED HINT STRATEGIES:
- **Playable hints**: Tell someone about a card they can play RIGHT NOW
- **Save hints**: Prevent discarding cards needed for sequences (especially 5s and critical cards)
- **Convention hints**: Use consistent hinting patterns your teammate can learn
- **Multi-card hints**: When possible, hint multiple cards at once
- **Bluff hints**: Sometimes hint a non-existent card to force a play

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
Analyze the game state and choose the most strategic action. Since you can see your own cards, use this advantage:

**DECISION FRAMEWORK:**
1. **Check for guaranteed plays**: Look at your hand - do you have any cards that are definitely playable?
2. **Assess teammate needs**: What cards do your teammates have that they could play with a hint?
3. **Evaluate discard safety**: Are any of your cards critical (like 5s) that teammates should know about?
4. **Token management**: Do you need info tokens? Can you afford to give hints?
5. **Endgame planning**: If deck is low, prioritize plays over hints

**SPECIFIC CONSIDERATIONS:**
- Since you can see your cards, you can make more confident plays
- Use hints to coordinate with teammates who can't see their own cards
- Don't waste hints on obvious plays - your teammates can see their cards too
- Focus on building sequences efficiently rather than just playing random cards
- Save critical cards (especially 5s) from being discarded

Choose the action that maximizes your team's chance of achieving a high score (aim for 20+ points). Be strategic and coordinate with your teammate!"""
        
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
            # Add current observation to history (before making decision)
            self._add_observation_to_history(observation)
            
            # Check rate limits before making API call
            self._check_rate_limit()
            
            # Create prompt
            prompt = self._create_prompt(observation)
            
            # Get response from Gemini with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    response_text = response.text
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str and "quota" in error_str.lower():
                        # For 429 errors, wait until the next minute
                        current_time = time.time()
                        seconds_until_next_minute = 60 - (current_time % 60) + 2  # Add 2 second buffer
                        
                        if attempt < max_retries - 1:  # Don't wait on last attempt
                            print(f"Rate limit hit (attempt {attempt + 1}/{max_retries}). Waiting {seconds_until_next_minute:.1f} seconds until next minute...")
                            time.sleep(seconds_until_next_minute)
                            # Reset our rate limiting tracking since we waited
                            self.request_times = []
                            self.current_minute = int(time.time() // 60)
                            continue
                        else:
                            raise e  # Re-raise on final attempt
                    else:
                        raise e  # Re-raise non-rate-limit errors immediately
            
            # Parse response
            action = self._parse_llm_response(response_text)
            
            if action is None:
                # Fallback to random action if parsing fails
                print(f"Failed to parse LLM response: {response_text}")
                fallback_action = random.choice(observation['legal_moves'])
                self._add_action_to_history(fallback_action)
                return fallback_action
                
            # Validate action is legal
            legal_moves = observation['legal_moves']
            for legal_move in legal_moves:
                if (action.get('action_type') == legal_move.get('action_type') and
                    action.get('card_index') == legal_move.get('card_index') and
                    action.get('color') == legal_move.get('color') and
                    action.get('rank') == legal_move.get('rank') and
                    action.get('target_offset') == legal_move.get('target_offset')):
                    # Add the action to history before returning
                    self._add_action_to_history(action)
                    return action
                    
            # If action is not legal, choose random legal action
            print(f"LLM action not legal: {action}")
            fallback_action = random.choice(observation['legal_moves'])
            self._add_action_to_history(fallback_action)
            return fallback_action
            
        except Exception as e:
            print(f"Error in Gemini agent: {e}")
            # Fallback to random action
            fallback_action = random.choice(observation['legal_moves'])
            self._add_action_to_history(fallback_action)
            return fallback_action
