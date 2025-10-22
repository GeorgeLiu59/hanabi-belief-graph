"""Prompt management module for Hanabi agents."""

import json
from typing import Dict, Any, List


class PromptManager:
    """Manages all prompt generation for Hanabi agents."""

    def get_hanabi_game_rules(self) -> str:
        """Hanabi game information."""
        return """## HANABI GAME INFORMATION

**DECK COMPOSITION:**
- 5 colors: red, blue, green, white, yellow
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
- **ABSOLUTE RULE**: Clued cards MUST have the hinted property
- **NEGATIVE INFERENCE**: Un-clued cards CANNOT have the hinted property
- **EXHAUSTIVE**: Clues affect ALL matching cards in the target player's hand

**CLUE MECHANICS:**
- When someone says "These cards are BLUE", they are telling the TRUTH
- All blue cards in that hand ARE marked by the clue
- All non-blue cards in that hand ARE NOT marked by the clue
- There is NO ambiguity or uncertainty in the clue system"""

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
- JSON keys follow the card numbering convention (e.g., `P2_Card3` means card number 3)
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
**CERTAINTY VARIANT UPDATE:**


"""
        elif variant == 'probabilistic':
            base_prompt += """
**PROBABILISTIC VARIANT UPDATE:**


"""
        elif variant == 'theory_of_mind':
            base_prompt += """
**THEORY OF MIND VARIANT UPDATE:**


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

**RESPONSE FORMAT:**
Return the updated belief graph inside a ```json``` block with no extra text in the block. After closing the block add a `REASONING:` section explaining the key updates.
Update existing structures IN PLACE - do not create new JSON blocks."""

        return base_prompt

    def get_rules_and_mechanics_block(self) -> str:
        """Core Hanabi rules and mechanics."""
        return """## HANABI RULES & MECHANICS
- **CRITICAL**: You CANNOT see your own cards! This is the fundamental rule of Hanabi.
- **CRITICAL**: You CAN see other players' cards, but they cannot see their own either.
- **CardKnowledge variant**: You get explicit information about what you've been told about your cards.
- Goal: Play cards in ascending order (1→2→3→4→5) for each color (R=Red, Y=Yellow, G=Green, W=White, B=Blue)
- You have 8 information tokens (for giving hints) and 3 life tokens
- Playing a wrong card costs 1 life token and the card goes to discard pile
- Game ends when you run out of life tokens, deck is empty, or complete all fireworks
- Maximum score is 25 points (5 colors × 5 ranks)

**Game mechanics**: You hold cards facing away from you, must rely on hints from teammates, and can only play cards you're confident about."""

    def get_strategy_and_priorities_block(self) -> str:
        """Game information."""
        return ""

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
        """Decision process information."""
        return ""

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

            # Add card information
            if knowledge.get('color') is not None and rank_info is not None and rank_info >= 0:
                current_firework = observation['fireworks'][knowledge['color']]
                if rank_info == current_firework:
                    if rank_info == 0:
                        game_state += " - this is a 1"
                    else:
                        game_state += " - playable"
                elif rank_info < current_firework:
                    game_state += " - already played"
                else:
                    game_state += f" - need {current_firework + 1} first"

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

        # Game state information
        game_state += "\n## GAME STATE\n"
        if observation['information_tokens'] == 0:
            game_state += "- No info tokens left\n"
        elif observation['information_tokens'] == 8:
            game_state += "- Max info tokens\n"

        # Card play information
        obvious_plays = []
        for i, (card, knowledge) in enumerate(zip(my_hand, my_knowledge), 1):
            color = knowledge.get('color')
            rank_info = knowledge.get('rank')
            if color is not None and rank_info is not None and rank_info >= 0:
                current_firework = observation['fireworks'][color]
                if rank_info == current_firework:
                    obvious_plays.append(f"Card {i} ({color}{rank_info + 1})")

        if obvious_plays:
            game_state += f"- Playable cards: {', '.join(obvious_plays)}\n"
        else:
            game_state += "- No playable cards\n"

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

        history_text += "\n**Pattern Analysis:**\n"

        # Analyze patterns in the history
        if len(action_history) >= 2:
            recent_actions = [action['action_type'] for action in action_history[-3:]]
            if recent_actions.count('REVEAL_COLOR') + recent_actions.count('REVEAL_RANK') >= 2:
                history_text += "- Recent hints given\n"
            elif recent_actions.count('PLAY') >= 2:
                history_text += "- Recent successful plays\n"
            elif recent_actions.count('DISCARD') >= 2:
                history_text += "- Recent discards\n"

        # Information token status
        if observation_history:
            current_tokens = observation_history[-1]['information_tokens']
            if current_tokens == 0:
                history_text += "- No information tokens remaining\n"
            elif current_tokens == 8:
                history_text += "- Maximum information tokens\n"

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

            # Use natural language version - must be available
            prompt_sections.extend([
                observation['belief_graph_natural_language'],
                "",
                "## BELIEF GRAPH SNAPSHOT (JSON)",
                "```json",
                json.dumps(observation['belief_graph'], indent=2),
                "```",
                ""
            ])

        prompt_sections.extend([
            self.get_format_requirements_block(),
            "",
            "## CURRENT GAME STATE:",
            game_state,
            "",
            "## DECISION REQUIRED:",
            "Choose your action."
        ])

        return "\n".join(prompt_sections)

    def create_retry_prompt(self, observation: Dict[str, Any], error_message: str) -> str:
        """Create a retry prompt with error feedback."""
        game_state = self.format_observation_for_llm(observation)

        retry_prompt = f"""PREVIOUS ACTION WAS ILLEGAL

{error_message}

Available legal moves:
"""
        for i, move in enumerate(observation['legal_moves']):
            retry_prompt += f"{i}: {move}\n"

        retry_prompt += f"""
{game_state}

Respond with a valid JSON action object:
{{"action_type": "...", "card_index": ..., "color": "...", "rank": ..., "target_offset": ...}}"""

        return retry_prompt

    def format_belief_graph_natural_language(self, belief_graph: Dict[str, Any], variant: str) -> str:
        """Format belief graph to natural language based on variant type."""
        if variant == 'certainty':
            return self.format_certainty_belief_graph_natural_language(belief_graph)
        elif variant == 'probabilistic':
            return self.format_probabilistic_belief_graph_natural_language(belief_graph)
        elif variant == 'theory_of_mind':
            return self.format_theory_of_mind_belief_graph_natural_language(belief_graph)
        else:
            # Fallback to certainty format if variant not recognized
            return self.format_certainty_belief_graph_natural_language(belief_graph)

    def format_certainty_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert certainty belief graph to natural language (pure information)."""
        nl_description = "## BELIEF GRAPH ANALYSIS (CERTAINTY VARIANT)\n\n"

        gs = belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"

        nl_description += "**MY HAND BELIEFS:**\n"

        game_state = belief_graph.get('GameState', {})
        fireworks = game_state.get('fireworks', {})
        if not fireworks:
            fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}

        for card_id, beliefs in belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            card_idx = int(card_num) - 1
            colors = beliefs['possible_colors']
            ranks = beliefs['possible_ranks']

            if len(colors) == 1 and len(ranks) == 1:
                color = colors[0]
                rank = ranks[0]
                color_key = color[0].upper() if len(color) > 0 else color.upper()
                current_firework = fireworks.get(color_key, 0)

                if rank == current_firework + 1:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - playable now\n"
                elif rank <= current_firework:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - already played\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - not yet\n"
            elif len(ranks) == 1 and ranks[0] == 1:
                possible_plays = []
                for c in colors:
                    c_key = c[0].upper() if len(c) > 0 else c.upper()
                    if fireworks.get(c_key, 0) == 0:
                        possible_plays.append(c.upper())

                if possible_plays:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): Rank 1, can start {'/'.join(possible_plays)}\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): Rank 1, colors already started\n"
            elif len(colors) == 1 and len(ranks) <= 2:
                color = colors[0]
                nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {'/'.join(map(str,ranks))}\n"
            elif len(colors) == 1:
                nl_description += f"- Card {card_num} (card_index {card_idx}): {colors[0].upper()} color only\n"
            elif len(ranks) == 1:
                nl_description += f"- Card {card_num} (card_index {card_idx}): Rank {ranks[0]} only\n"
            else:
                nl_description += f"- Card {card_num} (card_index {card_idx}): {len(colors)} colors, {len(ranks)} ranks possible\n"

        nl_description += "\n**TEAMMATE KNOWLEDGE MODEL:**\n"
        for player_hand, hand_data in belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            nl_description += f"\nPlayer {player_num}'s cards:\n"

            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')
                belief = card_data.get(f'p{player_num}_belief', {})
                colors = belief.get('possible_colors', [])
                ranks = belief.get('possible_ranks', [])

                if len(colors) == 1 and len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know {colors[0].upper()} {ranks[0]}\n"
                elif len(colors) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know color {colors[0].upper()}\n"
                elif len(ranks) == 1:
                    nl_description += f"  - Card {card_num}: I see {actual}, they know rank {ranks[0]}\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, uncertain\n"

        return nl_description

    def format_probabilistic_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert probabilistic belief graph to natural language (pure information)."""
        nl_description = "## BELIEF GRAPH ANALYSIS (PROBABILISTIC VARIANT)\n\n"

        gs = belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"

        nl_description += "**MY HAND BELIEFS:**\n"

        # Get current fireworks for playability checking
        game_state = belief_graph.get('GameState', {})
        fireworks = game_state.get('fireworks', {})
        if not fireworks:
            fireworks = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}

        for card_id, beliefs in belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            card_idx = int(card_num) - 1
            color_dist = beliefs['color_distribution']
            rank_dist = beliefs['rank_distribution']

            max_color = max(color_dist, key=color_dist.get)
            max_color_prob = color_dist[max_color]
            max_rank = max(rank_dist, key=rank_dist.get)
            max_rank_prob = rank_dist[max_rank]

            if max_color_prob == 1.0 and max_rank_prob == 1.0:
                # 100% certain - check if actually playable
                color = max_color
                rank = int(max_rank)
                color_key = color[0].upper() if len(color) > 0 else color.upper()
                current_firework = fireworks.get(color_key, 0)

                if rank == current_firework + 1:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - playable\n"
                elif rank <= current_firework:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - already played\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): {color.upper()} {rank} - waiting\n"
            elif max_color_prob >= 0.8 and max_rank_prob >= 0.8:
                # High confidence (80%+)
                nl_description += f"- Card {card_num} (card_index {card_idx}): {max_color.upper()} {max_rank} ({max_color_prob*100:.0f}%/{max_rank_prob*100:.0f}% confidence)\n"
            elif max_rank_prob >= 0.8 and max_rank == "1":
                # Very confident it's a rank 1
                possible_starts = []
                for c in color_dist:
                    c_key = c[0].upper() if len(c) > 0 else c.upper()
                    if fireworks.get(c_key, 0) == 0 and color_dist[c] > 0.5:
                        possible_starts.append(c.upper())

                if possible_starts:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): Likely rank 1 - can start {'/'.join(possible_starts)} ({max_rank_prob*100:.0f}% confidence)\n"
                else:
                    nl_description += f"- Card {card_num} (card_index {card_idx}): Likely rank 1 - {max_color.upper()} ({max_rank_prob*100:.0f}% confidence)\n"
            else:
                nl_description += f"- Card {card_num} (card_index {card_idx}): {max_color} {max_rank} ({max_color_prob*100:.0f}%/{max_rank_prob*100:.0f}% confidence)\n"

        nl_description += "\n**TEAMMATE KNOWLEDGE MODEL:**\n"
        for player_hand, hand_data in belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            nl_description += f"\nPlayer {player_num}'s cards:\n"

            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')
                belief = card_data.get(f'p{player_num}_belief', {})
                color_dist = belief.get('color_distribution', {})
                rank_dist = belief.get('rank_distribution', {})

                max_color_prob = max(color_dist.values()) if color_dist else 0
                max_rank_prob = max(rank_dist.values()) if rank_dist else 0

                if max_color_prob == 1.0 and max_rank_prob == 1.0:
                    nl_description += f"  - Card {card_num}: I see {actual}, certain knowledge\n"
                elif max_color_prob >= 0.8:
                    nl_description += f"  - Card {card_num}: I see {actual}, color known ({max_color_prob*100:.0f}%)\n"
                elif max_rank_prob >= 0.8:
                    nl_description += f"  - Card {card_num}: I see {actual}, rank known ({max_rank_prob*100:.0f}%)\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, uncertain\n"

        return nl_description

    def format_theory_of_mind_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert Theory of Mind belief graph to natural language (pure information)."""
        nl_description = "## BELIEF GRAPH ANALYSIS (THEORY OF MIND VARIANT)\n\n"

        gs = belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"

        tom = belief_graph['ToM_Layer']
        nl_description += "**TEAM DYNAMICS ANALYSIS:**\n"

        team_focus = tom['Team_Focus']['focus_distribution']
        max_focus = max(team_focus, key=team_focus.get)
        focus_prob = team_focus[max_focus]
        nl_description += f"- Team Strategy: {max_focus} ({focus_prob*100:.0f}% confidence)\n"

        nl_description += "\n**TEAMMATE PROFILES:**\n"
        for player, profile in tom['Teammates'].items():
            skill = profile['inferred_skill']
            aggr = profile.get('play_aggressiveness', 0.5)
            hint_q = profile.get('hint_quality', 0.5)

            skill_desc = "expert" if skill > 0.8 else "competent" if skill > 0.6 else "average" if skill > 0.4 else "novice"
            aggr_desc = "aggressive" if aggr > 0.7 else "balanced" if aggr > 0.3 else "conservative"
            hint_desc = "excellent hints" if hint_q > 0.8 else "good hints" if hint_q > 0.6 else "average hints" if hint_q > 0.4 else "poor hints"

            nl_description += f"- {player}: {skill_desc} player (skill: {skill:.1f}), {aggr_desc} style, {hint_desc}\n"

        nl_description += "\n**CARD BELIEFS (probabilistic base):**\n"

        nl_description += "\nMy Hand:\n"
        for card_id, beliefs in belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            color_dist = beliefs['color_distribution']
            rank_dist = beliefs['rank_distribution']

            max_color = max(color_dist, key=color_dist.get)
            max_rank = max(rank_dist, key=rank_dist.get)

            nl_description += f"- Card {card_num}: Likely {max_color} {max_rank}\n"

        nl_description += "\n**TEAMMATE HANDS (with mental models):**\n"
        for player_hand, hand_data in belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            player_key = f"P{player_num}"

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
