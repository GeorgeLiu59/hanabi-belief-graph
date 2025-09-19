"""Prompt management module for Hanabi agents."""

from typing import Dict, Any, List


class PromptManager:
    """Manages all prompt generation for Hanabi agents."""
    
    def get_rules_and_mechanics_block(self) -> str:
        """Core Hanabi rules and unique mechanics."""
        return """## HANABI RULES & MECHANICS
- **CRITICAL**: You CANNOT see your own cards! This is the fundamental rule of Hanabi.
- **CRITICAL**: You CAN see other players' cards, but they cannot see their own either.
- **CardKnowledge variant**: You get explicit information about what you've been told about your cards.
- Goal: Play cards in ascending order (1→2→3→4→5) for each color (R=Red, Y=Yellow, G=Green, W=White, B=Blue)
- You have 8 information tokens (for giving hints) and 3 life tokens
- Playing a wrong card costs 1 life token and the card goes to discard pile
- Game ends when you run out of life tokens, deck is empty, or complete all fireworks
- Maximum score is 25 points (5 colors × 5 ranks)

**Why this matters**: You hold cards facing away from you, must rely on hints from teammates, and can only play cards you're confident about. The tension is between giving helpful hints vs. making safe plays."""

    def get_strategy_and_priorities_block(self) -> str:
        """Strategic guidelines, action priorities, and hint techniques."""
        return """## STRATEGY & PRIORITIES

**Core Principles:**
1. **AGGRESSIVE EARLY GAME**: Play 1s immediately - they're always safe and start sequences
2. **CONFIDENCE PLAYING**: Only play cards you're 100% certain about (except 1s)
3. **HINT EFFICIENCY**: Give hints that create immediate plays or save critical cards
4. **TOKEN MANAGEMENT**: Balance information tokens - don't waste them on obvious plays
5. **SEQUENCE BUILDING**: Focus on completing one color at a time when possible
6. **ENDGAME AWARENESS**: When deck is low, prioritize plays over hints

**Action Priority (in order):**
1. **PLAY** any 1 you know about - they're always safe and essential for progress
2. **PLAY** a card you're certain is the next needed card (highest priority)
3. **HINT** to enable a teammate's immediate play (especially 1s they can play)
4. **HINT** to save critical cards from being discarded (especially 5s)
5. **DISCARD** when you need info tokens and have no better options
6. **DISCARD** obviously useless cards (duplicates, too-low ranks)

**Critical Early Game Strategy:**
- **ALWAYS PLAY 1s**: If you know a card is a 1, play it immediately - it's never wrong
- **HINT ABOUT 1s**: If you see a teammate has a 1, hint about it immediately
- **BUILD MOMENTUM**: Getting even one firework started opens up more opportunities

**Hint Techniques:**
- **Playable hints**: Tell someone about a card they can play RIGHT NOW (especially 1s)
- **Save hints**: Prevent discarding cards needed for sequences (especially 5s)
- **Multi-card hints**: When possible, hint multiple cards at once
- **1s are special**: Always prioritize hinting about 1s - they're free points"""

    def get_format_requirements_block(self) -> str:
        """JSON response format specification."""
        return """## FORMAT REQUIREMENTS:
Respond with ONLY a valid JSON object in this exact format:
{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 0-4, "target_offset": 1-3}

**IMPORTANT**: 
- For PLAY/DISCARD: use "card_index" (0-4), set color/rank/target_offset to null
- For REVEAL_COLOR: use "color" and "target_offset", set card_index/rank to null  
- For REVEAL_RANK: use "rank" and "target_offset", set card_index/color to null"""

    def get_decision_framework_block(self) -> str:
        """Complete decision-making process and tactical considerations."""
        return """## DECISION FRAMEWORK:

**Step-by-step process:**
1. **Check your knowledge**: Based on hints you've received, do you know any cards you can safely play?
2. **Assess teammate needs**: What cards do your teammates have that they could play with a hint?
3. **Give helpful hints**: Can you tell teammates about playable cards or save critical cards?
4. **Safe discarding**: If you must discard, choose cards that are least likely to be important
5. **Token management**: Balance between giving hints and making plays

**Key considerations:**
- You can only play cards you're certain about based on hints received
- Give hints to help teammates identify their playable cards
- Look at teammates' hands - hint about cards they can play immediately
- Save critical cards (especially 5s and cards needed for sequences) from being discarded
- When unsure, giving a helpful hint is often safer than guessing a play
- Pay attention to what cards are in the discard pile - some sequences may be impossible now"""

    def format_observation_for_llm(self, observation: Dict[str, Any]) -> str:
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
            playable_cards = []
            for j, card in enumerate(hand):
                if card['color'] is not None and card['rank'] is not None:
                    current_firework = observation['fireworks'][card['color']]
                    if card['rank'] == 0:  # It's a 1
                        if current_firework == 0:
                            game_state += f"🎯{card['color']}1(HINT!) "
                            playable_cards.append(f"1 at position {j}")
                        else:
                            game_state += f"{card['color']}{card['rank']} "
                    elif card['rank'] == current_firework:
                        game_state += f"✅{card['color']}{card['rank']}(HINT!) "
                        playable_cards.append(f"{card['color']}{card['rank']} at position {j}")
                    else:
                        game_state += f"{card['color']}{card['rank']} "
                else:
                    game_state += "? "
            if playable_cards:
                game_state += f" <- CAN PLAY: {', '.join(playable_cards)}"
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
                if knowledge['rank'] == 0:  # It's a 1
                    if current_firework == 0:
                        game_state += " 🎯 THIS IS A 1 - PLAY IT NOW! (Always safe)"
                    else:
                        game_state += " ❌ This 1 already played"
                elif knowledge['rank'] == current_firework:
                    game_state += " ✅ PLAYABLE NOW!"
                elif knowledge['rank'] < current_firework:
                    game_state += " ❌ Too low - already played"
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

    def format_history_for_llm(self, observation_history: List[Dict], action_history: List[Dict]) -> str:
        """Format the observation and action history for the LLM."""
        if not observation_history:
            return "## GAME HISTORY: This is the first turn of the game.\n"
        
        history_text = "## GAME HISTORY (Previous Turns)\n"
        
        for i, (obs, action) in enumerate(zip(observation_history, action_history)):
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
                prev_obs = observation_history[i-1]
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
        if len(action_history) >= 2:
            recent_actions = [action['action_type'] for action in action_history[-3:]]
            if recent_actions.count('REVEAL_COLOR') + recent_actions.count('REVEAL_RANK') >= 2:
                history_text += "- You've been giving many hints recently - consider playing cards or discarding\n"
            elif recent_actions.count('PLAY') >= 2:
                history_text += "- You've been playing cards successfully - good progress!\n"
            elif recent_actions.count('DISCARD') >= 2:
                history_text += "- You've been discarding frequently - make sure you're not discarding important cards\n"
        
        # Check for information token management
        if observation_history:
            current_tokens = observation_history[-1]['information_tokens']
            if current_tokens == 0:
                history_text += "- ⚠️ No information tokens left - must discard to get more\n"
            elif current_tokens == 8:
                history_text += "- ⚠️ Maximum information tokens - should give hints or play cards\n"
        
        return history_text

    def create_main_prompt(self, observation: Dict[str, Any], history: str) -> str:
        """Create the main prompt for the LLM using modular building blocks."""
        game_state = self.format_observation_for_llm(observation)
        
        # Build prompt using consolidated modular components
        prompt_sections = [
            "You are playing Hanabi-Full-CardKnowledge, a cooperative card game where you must work with your teammates to build fireworks.",
            "",
            history,
            "",
            self.get_rules_and_mechanics_block(),
            "",
            self.get_strategy_and_priorities_block(),
            "",
            self.get_format_requirements_block(),
            "",
            "## CURRENT GAME STATE:",
            game_state,
            "",
            "## YOUR TASK:",
            "Analyze the game state and choose the most strategic action. Remember: you CANNOT see your own cards!",
            "",
            self.get_decision_framework_block(),
            "",
            "Choose the action that maximizes your team's chance of achieving a high score (aim for 20+ points). Be strategic and coordinate with your teammate!"
        ]
        
        return "\n".join(prompt_sections)

    def create_retry_prompt(self, observation: Dict[str, Any], error_message: str) -> str:
        """Create a retry prompt with error feedback."""
        game_state = self.format_observation_for_llm(observation)
        
        retry_prompt = f"""PREVIOUS ACTION WAS ILLEGAL!

{error_message}

You must choose from ONLY the legal moves listed below. Do not make up actions.

## CURRENT LEGAL MOVES:
"""
        for i, move in enumerate(observation['legal_moves']):
            retry_prompt += f"{i}: {move}\n"
            
        retry_prompt += f"""
{game_state}

Respond with ONLY a valid JSON object that matches EXACTLY one of the legal moves above:
{{"action_type": "...", "card_index": ..., "color": "...", "rank": ..., "target_offset": ...}}"""
        
        return retry_prompt