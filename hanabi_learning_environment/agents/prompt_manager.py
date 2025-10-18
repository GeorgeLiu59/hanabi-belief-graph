"""Prompt management module for Hanabi agents."""

import json
from typing import Dict, Any, List


class PromptManager:
    """Manages all prompt generation for Hanabi agents."""

    def get_hanabi_game_rules(self) -> str:
        """Complete Hanabi game rules for intelligent LLM reasoning."""
        return """## HANABI COMPLETE GAME RULES

**DECK COMPOSITION:**
- 5 colors: red, blue, green, white, yellow, yellow
- 5 ranks: 1, 2, 3, 4, 5
- Each combination appears exactly once (25 unique cards total)
- Distribution: Three 1s, two each of 2-4, one 5 per color

**OBJECTIVE:**
- Play cards in ascending order by color (1→2→3→4→5) to build fireworks
- Successfully playing all 5 cards of a color scores 5 points
- Maximum score: 25 points (5 colors × 5 ranks)
- Playing a 5 earns an extra clue token

**CLUE SYSTEM:**
- **Color clues**: "These cards are [color]" - affects ALL cards of that color in hand
- **Rank clues**: "These cards are [rank]" - affects ALL cards of that rank in hand
- **ABSOLUTE RULE**: Clued cards MUST have the hinted property - NO EXCEPTIONS
- **NEGATIVE INFERENCE**: Un-clued cards CANNOT have the hinted property - DEFINITIVE
- **EXHAUSTIVE**: Clues affect ALL matching cards in the target player's hand

**CLUE MECHANICS ARE ABSOLUTE:**
- When someone says "These cards are BLUE", they are telling the TRUTH
- All blue cards in that hand ARE marked by the clue
- All non-blue cards in that hand ARE NOT marked by the clue
- This is the fundamental rule of Hanabi communication
- There is NO ambiguity or uncertainty in the clue system

**STRATEGIC INSIGHTS:**
- **Early game**: Color clues often suggest rank 1 (most common card)
- **Late game**: Rank 5 clues are very informative (only one per color)
- **Play timing**: Clues reveal player skill and team coordination
- **Card value**: Some cards are "trash" (duplicates, unplayable sequences)
- **Probability**: Consider what's been played and remaining deck composition"""

    def get_belief_update_prompt(self, event: Dict[str, Any], current_beliefs: Dict[str, Any] = None, variant: str = 'certainty') -> str:
        """Generate intelligent belief update prompt based on event and variant type."""
        clue_type = event.get('clue_type')
        raw_value = event.get('value')
        clued_value = raw_value
        if clue_type == 'rank':
            int_value = None
            if raw_value is not None:
                try:
                    int_value = int(raw_value)
                except (TypeError, ValueError):
                    int_value = None
            if int_value is not None:
                clued_value = int_value + 1
        clued_indices = event.get('card_indices', [])
        clued_cards = [i + 1 for i in clued_indices]
        recipient = event.get('clue_recipient', 'Unknown')
        giver = event.get('clue_giver', 'Unknown')

        # Base rules section - reused across all variants
        base_prompt = f"""{self.get_hanabi_game_rules()}

**CURRENT CLUE EVENT:**
- Clue giver: {giver}
- Clue recipient: {recipient}
- Clue type: {clue_type}
- Clue value: {clued_value}
- Target card numbers (leftmost = 1): {clued_cards}

**BELIEF UPDATE REQUIREMENTS:**
- Cards numbered {clued_cards} ARE {clued_value}
- Cards with other numbers ARE NOT {clued_value}
- Update beliefs using your Hanabi expertise and strategic reasoning
- IMPORTANT: JSON keys already follow the card numbering convention (e.g., `P2_Card3` means card number 3)
"""

        recipient_section = """
**HAND SECTION TO UPDATE:**
- Clue recipient: {recipient}
- If {recipient} is YOU, update the cards in `My_Hand_Beliefs` using keys `P{recipient_num}_CardN`.
- If {recipient} is a teammate, update only `Teammate_Hand_Beliefs["P{recipient_num}_Hand"]["P{recipient_num}_CardN"]`.
- Do NOT touch any other player's belief sections or rename structures.
"""

        if recipient.startswith('P') and recipient[1:].isdigit():
            recipient_num = int(recipient[1:])
            base_prompt += recipient_section.format(recipient=recipient, recipient_num=recipient_num)
        else:
            base_prompt += """
**HAND SECTION TO UPDATE:**
- Apply the clue ONLY to the recipient's existing belief entries.
- Never modify other players' sections or rename structures.
"""

        # Variant-specific sections
        if variant == 'certainty':
            base_prompt += """
**CERTAINTY VARIANT TASK:**
Update your belief graph by eliminating impossible combinations and narrowing possibilities:
- Set clued cards to ONLY {clued_value} for {clue_type}
- Remove {clued_value} from un-clued cards for {clue_type}
- Maintain all other possibilities that remain viable
- Use strategic reasoning about what this clue reveals

"""
        elif variant == 'probabilistic':
            base_prompt += """
**PROBABILISTIC VARIANT TASK:**
Update probability distributions with intelligent reasoning:
- Clued cards: Increase probability of {clued_value} (potentially to 100%)
- Un-clued cards: Set {clued_value} probability to 0%, renormalize others
- Consider strategic probability factors:
  * Early color clues → higher rank 1 probability
  * Late game rank clues → very informative
  * Deck composition and what's been played
  * Clue timing reveals team strategy and player skill level

"""
        elif variant == 'theory_of_mind':
            base_prompt += """
**THEORY OF MIND VARIANT TASK:**
Multi-level analysis required:

1. **Card Beliefs**: Update beliefs about actual cards (same as certainty variant)
2. **Player Modeling**: What does this clue reveal about:
   - The giver's skill level and Hanabi expertise?
   - Team strategy (aggressive vs conservative play)?
   - Is this a "save" clue (protecting important cards) or "play" clue (enabling plays)?
3. **Team Strategy**: How does this affect:
   - Team coordination and communication patterns?
   - Strategic priorities for current game state?
   - What does this reveal about team psychology?

Use your understanding of Hanabi psychology to update both card beliefs AND Theory of Mind models.
"""

        # Add current beliefs if provided
        if current_beliefs:
            base_prompt += (
                "\nCRITICAL JSON UPDATE INSTRUCTIONS:\n"
                "  - You MUST UPDATE the existing belief structures in place\n"
                "  - DO NOT create new JSON blocks like \"Updated_P2_Hand_Beliefs\"\n"
                "  - Find each player's belief section in the Current Beliefs and MODIFY it directly\n"
                "  - For P2_Hand_Beliefs, modify that exact structure\n"
                "  - For P3_Hand_Beliefs, modify that exact structure\n"
                "  - For P4_Hand_Beliefs, modify that exact structure\n"
                "  - Keep the same structure names and organization\n"
                "  - Only update the values within existing arrays and objects\n"
                "  - Your output should be the complete updated Current Beliefs JSON structure\n"
                "\nCurrent Beliefs:\n"
                f"{json.dumps(current_beliefs, indent=2)}"
            )

        # Final instruction
        base_prompt += """

**INSTRUCTION:**
Return the updated belief graph inside a ```json``` block with no extra text in the block. After closing the block add a `REASONING:` section (it can span multiple lines) explaining the key updates.
Focus on intelligent inference, not just mechanical rule application.
Apply your complete understanding of Hanabi strategy and player psychology.
Remember to update existing structures IN PLACE - do not create new JSON blocks."""

        return base_prompt

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
1. **PLAY** cards marked ✅ CERTAIN - you know BOTH color AND rank with 100% certainty
2. **PLAY** cards marked ⚠️ SAFE BET - special cases where risk is acceptable:
   - Rank 1 + known color (1s are always playable if that color hasn't started)
   - Known rank + highly probable color (only 1-2 color possibilities left)
3. **HINT** to enable a teammate's immediate play (especially helping them identify 1s)
4. **HINT** to save critical cards from being discarded (especially 5s)
5. **DISCARD** when you need info tokens and have no better options
6. **AVOID** playing cards marked ❌ RISKY - too much uncertainty

**Critical Early Game Strategy:**
- **1s ARE SPECIAL**: If you know a card is rank 1 (even if color uncertain), it's often worth playing - 1s start all sequences
- **HINT EFFICIENTLY**: One good hint can help identify multiple cards through negative inference
- **BUILD MOMENTUM**: Early plays create more options - don't get stuck in hint loops
- **ACCEPTABLE RISKS**: With 3 lives, taking a calculated risk on a likely 1 is often better than endless hinting

**Hint Techniques:**
- **Playable hints**: Tell someone about a card they can play RIGHT NOW (especially 1s)
- **Save hints**: Prevent discarding cards needed for sequences (especially 5s)
- **Multi-card hints**: When possible, hint multiple cards at once
- **1s are special**: Always prioritize hinting about 1s - they're free points"""

    def get_format_requirements_block(self) -> str:
        """JSON response format specification."""
        return """## FORMAT REQUIREMENTS:
Respond with ONLY a valid JSON object in this exact format (ranks MUST use human numbers 1-5):
{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 1-5, "target_offset": 1-3}

**IMPORTANT**:
- For PLAY/DISCARD: use "card_index" (0-4), set color/rank/target_offset to null
- For REVEAL_COLOR: use "color" and "target_offset", set card_index/rank to null
- For REVEAL_RANK: use "rank" (1-5) and "target_offset", set card_index/color to null"""

    def get_decision_framework_block(self) -> str:
        """Complete decision-making process and tactical considerations."""
        return """## DECISION FRAMEWORK:

**Step-by-step process:**
1. **CHECK FOR CERTAIN PLAYS**: Do you have any ✅ CERTAIN cards? Play these immediately!
2. **CHECK FOR SAFE BETS**: Do you have ⚠️ SAFE BET cards (especially rank 1s)? These are worth playing if tokens are low or progress is needed
3. **ASSESS TEAMMATE NEEDS**: Can you give a hint that enables an immediate play?
4. **STRATEGIC HINTS**: Save critical cards (5s) or help narrow down possibilities
5. **DISCARD WISELY**: When stuck, discard the riskiest/least useful cards
6. **BALANCE TOKENS**: Don't let info tokens max out (8/8) - that's wasteful

**Key considerations:**
- **PRIORITIZE CERTAINTY**: ✅ CERTAIN cards should always be played
- **CALCULATED RISKS**: ⚠️ SAFE BET cards (rank 1s with known color) are usually worth it
- **VERIFY PLAYABILITY**: Before playing, check that the rank matches what fireworks need next
- **DON'T WASTE HINTS**: If teammate already knows enough to play, don't redundantly hint
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
        current_player = observation.get('current_player')
        num_players = observation.get('num_players', len(observation['observed_hands']))

        for offset, hand in enumerate(observation['observed_hands'][1:], 1):
            actual_player = None
            if current_player is not None and num_players:
                actual_player = (current_player + offset) % num_players
            display_player = (actual_player + 1) if actual_player is not None else (offset + 1)
            game_state += f"Player {display_player}: "
            playable_cards = []
            for j, card in enumerate(hand, 1):
                color = card.get('color')
                rank = card.get('rank')
                if color is not None and rank is not None and rank >= 0:
                    display_rank = rank + 1
                    display_card = f"{color}{display_rank}"
                    current_firework = observation['fireworks'][color]
                    if rank == current_firework:
                        symbol = "🎯" if rank == 0 else "✅"
                        game_state += f"{symbol}{display_card}(HINT!) "
                        playable_cards.append(f"{display_card} at position {j}")
                    else:
                        game_state += f"{display_card} "
                else:
                    game_state += "? "
            if playable_cards:
                game_state += f" <- CAN PLAY: {', '.join(playable_cards)}"
            game_state += "\n"

        # Your hand with knowledge analysis
        game_state += "\n## YOUR HAND (what you know vs. what you need)\n"
        my_hand = observation['observed_hands'][0]
        my_knowledge = observation['card_knowledge'][0]

        for i, (card, knowledge) in enumerate(zip(my_hand, my_knowledge), 1):
            game_state += f"Card {i}: "
            if knowledge.get('color') is not None:
                game_state += f"Color: {knowledge['color']}, "
            else:
                game_state += "Color: unknown, "

            rank_info = knowledge.get('rank')
            if rank_info is not None and rank_info >= 0:
                game_state += f"Rank: {rank_info + 1}"
            else:
                game_state += "Rank: unknown"

            # Add strategic analysis
            if knowledge.get('color') is not None and rank_info is not None and rank_info >= 0:
                current_firework = observation['fireworks'][knowledge['color']]
                if rank_info == current_firework:
                    if rank_info == 0:
                        game_state += " 🎯 THIS IS A 1 - PLAY IT NOW! (Always safe)"
                    else:
                        game_state += " ✅ PLAYABLE NOW!"
                elif rank_info < current_firework:
                    game_state += " ❌ Too low - already played"
                else:
                    game_state += f" ⏳ Need {current_firework + 1} first"

            game_state += "\n"

        # Legal moves with strategic context
        game_state += "\n## AVAILABLE ACTIONS\n"
        for i, move in enumerate(observation['legal_moves']):
            if move['action_type'] == 'PLAY':
                game_state += f"{i}: Play card {move['card_index']}\n"
            elif move['action_type'] == 'DISCARD':
                game_state += f"{i}: Discard card {move['card_index']} (+1 info token)\n"
            elif move['action_type'] == 'REVEAL_COLOR':
                target_offset = move['target_offset']
                if current_player is not None and num_players:
                    target_player = (current_player + target_offset) % num_players
                    display_target = target_player + 1
                else:
                    display_target = target_offset
                game_state += f"{i}: Hint color {move['color']} to player {display_target} (-1 info token)\n"
            elif move['action_type'] == 'REVEAL_RANK':
                target_offset = move['target_offset']
                if current_player is not None and num_players:
                    target_player = (current_player + target_offset) % num_players
                    display_target = target_player + 1
                else:
                    display_target = target_offset
                game_state += f"{i}: Hint rank {move['rank'] + 1} to player {display_target} (-1 info token)\n"

        # Strategic recommendations
        game_state += "\n## STRATEGIC CONSIDERATIONS\n"
        if observation['information_tokens'] == 0:
            game_state += "- ⚠️ No info tokens left - must discard to get more\n"
        elif observation['information_tokens'] == 8:
            game_state += "- ⚠️ Max info tokens - should give hints or play cards\n"

        # Check for obvious plays
        obvious_plays = []
        for i, (card, knowledge) in enumerate(zip(my_hand, my_knowledge), 1):
            color = knowledge.get('color')
            rank_info = knowledge.get('rank')
            if color is not None and rank_info is not None and rank_info >= 0:
                current_firework = observation['fireworks'][color]
                if rank_info == current_firework:
                    obvious_plays.append(f"Card {i} ({color}{rank_info + 1})")

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
                history_text += f"- You gave a RANK hint ({action['rank'] + 1}) to player {action['target_offset']}\n"

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
            ""
        ]

        # Add belief graph data if present
        if 'belief_graph' in observation:
            belief_variant = observation['belief_variant']

            # Use natural language version - must be available
            prompt_sections.extend([
                observation['belief_graph_natural_language'],
                "",
                "## BELIEF GRAPH SNAPSHOT (JSON)",
                "```json",
                json.dumps(observation['belief_graph'], indent=2),
                "```",
                "",
                "Use this belief analysis to make more informed decisions about:",
                "- Which cards you can safely play based on your certainty",
                "- Which cards your teammates know about and can play",
                "- What hints would be most helpful to give",
                "- Which cards are safe to discard",
                ""
            ])

        prompt_sections.extend([
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
        ])

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
