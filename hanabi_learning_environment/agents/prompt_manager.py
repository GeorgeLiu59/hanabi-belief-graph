"""Prompt management module for Hanabi agents."""

import json
from typing import Dict, Any, List


class PromptManager:
    """Manages all prompt generation for Hanabi agents."""

    def _get_default_fireworks(self) -> Dict[str, int]:
        """Get default fireworks structure."""
        return {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}

    def get_hanabi_game_rules(self) -> str:
        """Hanabi game information."""
        return self.get_rules_and_mechanics_block()

    def get_belief_update_prompt(self, event: Dict[str, Any], current_beliefs: Dict[str, Any] = None, variant: str = 'certainty', history: str = "", observation: Dict[str, Any] = None, cot_reasoner=None) -> str:
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
        base_prompt = f"""You are playing Hanabi-Full-CardKnowledge.

{self.get_hanabi_game_rules()}

**COOPERATIVE TEAM CONTEXT:**
You and your teammates are working together to build fireworks and score the maximum 25 points. This is a COOPERATIVE game - your success depends on teamwork.

**GAME HISTORY:**
{history}

**CURRENT GAME STATE:**
{self.format_observation_for_llm(observation)}

**CURRENT CLUE EVENT:**
- Clue giver: {giver}
- Clue recipient: {recipient}
- Clue type: {clue_type}
- Clue value: {clued_value}
- Target card numbers (leftmost = 1): {clued_cards}

**GENERAL BELIEF GRAPH UPDATE PRMPT - STRATEGIC TEAM REASONING:**
Think cooperatively about this clue event:
- What game stage is this? (early/mid/late game based on remaining cards, clues, lives)
- What is the team's current priority? (building fireworks, saving important cards, managing information)
- What is {giver} trying to communicate? (which card to play/save, signaling strategy)
- How does this help {recipient} make better decisions?
- How should this update affect the team's shared mental model?

**BELIEF UPDATE REQUIREMENTS:**
- Update beliefs based on cooperative reasoning and team strategy
- Cards numbered {clued_cards} ARE {clued_value} (100% certain)
- Cards with other numbers ARE NOT {clued_value} (negative inference)
- Consider what this means for future team coordination
- Update mental models of what teammates know about their cards and the game state
- JSON keys follow the card numbering convention (e.g., `P2_Card3` means card number 3)
"""
        recipient_section = """
**HAND SECTION TO UPDATE:**
- Clue recipient: {recipient}
- If {recipient} is YOU, update your beliefs about your own cards in `My_Hand_Beliefs`
- If {recipient} is a teammate, update beliefs about their cards in `Teammate_Hand_Beliefs`
- Use your creativity to structure the belief updates in the most logical and helpful way
- Focus on accurately representing what the clue reveals and what can be inferred
- You can reorganize or restructure beliefs as needed to best reflect the new information
"""

        if recipient.startswith('P') and recipient[1:].isdigit():
            recipient_num = int(recipient[1:])
            base_prompt += recipient_section.format(recipient=recipient, recipient_num=recipient_num)
        else:
            base_prompt += recipient_section.format(recipient=recipient)

        # Variant-specific sections
        if variant == 'certainty':
            base_prompt += """
**CERTAINTY VARIANT UPDATE:**


"""
        elif variant in ['probabilistic', 'theory_of_mind']:
            base_prompt += """
**PROBABILISTIC VARIANT UPDATE:**


"""
        if variant == 'theory_of_mind':
            base_prompt += """
**THEORY OF MIND VARIANT UPDATE (Variation 4 - Recursive ToM):**
CRITICAL: You must perform recursive Theory of Mind reasoning about what teammates think about YOU.

**TEAM FOCUS STRATEGIES - DEFINITIONS AND INDICATORS:**

**PLAY_ANYTHING:**
- Meaning: No specific priority, play whatever seems most valuable
- Use when: Early game, multiple options available, no clear strategic priority
- Shift away from: When specific strategic needs emerge (save 5s, play 1s)

**SAVE_FIVES:**
- Meaning: Prioritize identifying and saving 5s for endgame (+1 clue token bonus)
- Use when: Mid-to-late game, clues are plentiful, 5s become valuable
- Indicators: Players giving conservative clues, holding potentially valuable cards

**PLAY_ONES:**
- Meaning: Prioritize playing 1s to start new fireworks (foundational strategy)
- Use when: Early game, multiple colors unstarted, high information-to-clue ratio
- Indicators: Players getting rank 1 clues, need to establish color foundations

**RECURSIVE ToM REASONING:**
1. **First-order ToM**: Update what you think about the clue recipient's beliefs about their own cards
2. **Second-order ToM (Variation 4)**: Update what you think the clue giver thinks about YOU (player n itself)
3. **Strategic Intent**: Analyze how this clue affects what everyone thinks about your capabilities and team strategy

**TEAM MENTAL MODEL UPDATES:**
- Update Team_Focus distribution based on strategic implications of the clue
- Consider GAME STAGE (early/mid/late) when determining appropriate focus
- Update inferred_skill for players based on the quality/complexity of their actions
- Update Recursive_ToM beliefs about what players think about YOUR skill and strategy (not about other players)
- Consider how this clue changes what others think about your decision-making capabilities
- CLUE TYPE ANALYSIS: Rank 1 clues often signal PLAY_ONES, complex clues may signal SAVE_FIVES strategy

**CRITICAL RECURSIVE ToM STRUCTURE:**
For each player's Recursive_ToM section about YOU (e.g., "P1 thinks about ME"), maintain this exact structure:
```json
"believes_about_my_hand": {
  "Card1": {"color": "R", "rank": 1, "confidence": 0.8},
  "Card2": {"color": "Y", "rank": 2, "confidence": 0.6}
},
"believes_about_my_skill": 0.55,
"believes_about_my_strategy": 0.6
```

- `believes_about_my_hand`: Update what this player thinks about your specific cards
- `believes_about_my_skill`: Update what this player thinks about your overall skill level (0-1)
- `believes_about_my_strategy`: Update what this player thinks about your strategic tendencies (0-1)


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
Return the updated belief graph inside a ```json``` block with no extra text in the block. After closing the block add a comprehensive `REASONING:` section explaining your belief update analysis.

**NOTE:** Your belief updates should follow Chain of Thought reasoning covering:
- Direct impact of the clue on card knowledge
- Team mental model and strategy updates
- Theory of Mind reasoning (for ToM variants)
- Strategic implications for future coordination

Update existing structures IN PLACE - do not create new JSON blocks."""

        # Apply Chain of Thought enhancement if CoT reasoner is provided
        if cot_reasoner:
            base_prompt = cot_reasoner.create_belief_update_cot_prompt(observation, base_prompt)

        return base_prompt

    def get_rules_and_mechanics_block(self) -> str:
        """Core Hanabi rules and mechanics - single source of truth."""
        return """## HANABI RULES & MECHANICS

**GAME SETUP:**
- 5 colors (R,G,W,Y,B) × 5 ranks (1-5) = 25 unique cards
- 3 ones, 2 each of 2-4, 1 five per color

**OBJECTIVE:**
- Build fireworks: play 1→2→3→4→5 for each color
- Score: 25 points max (5 colors × 5 ranks)
- Playing a 5 earns +1 clue token

**FUNDAMENTAL RULE:**
- You CANNOT see your own cards
- You CAN see everyone else's cards
- CardKnowledge variant: explicit info about clues you receive

**CLUE SYSTEM:**
- 8 information tokens for hints, 3 life tokens
- Color clues: "These cards are [COLOR]" - affects ALL cards of that color
- Rank clues: "These cards are [RANK]" - affects ALL cards of that rank
- ABSOLUTE: Clued cards ALWAYS have the hinted property
- NEGATIVE: Un-clued cards NEVER have the hinted property
- NO ambiguity - clues are 100% truthful

**GAMEPLAY:**
- Wrong play costs 1 life, card goes to discard
- Game ends: 0 lives, empty deck, or complete fireworks
- You hold cards facing away, rely on teammate hints"""

    def get_belief_graph_usage_hint(self, variant: str) -> str:
        """Get variant-specific hint on how to leverage the belief graph for better decisions."""

        if variant == 'certainty':
            return """
**BELIEF GRAPH INTELLIGENCE (CERTAINTY VARIANT):**
Your belief graph contains pre-computed logical constraints from the complete game history.

**Available Information:**
- Certain card identities (single remaining color + rank combinations)
- Negative constraints (colors/ranks definitively ruled out)
- Teammate knowledge states about their own cards
- Complete elimination possibilities

**Strategic Applications:**
- Certain cards can be played immediately when fireworks align
- Negative constraints enable safe discards of impossible cards
- Teammate knowledge states indicate optimal timing for clues
- Process of elimination reveals card identities through constraint satisfaction
"""

        result = ""
        if variant in ['probabilistic', 'theory_of_mind']:
            result += """
**BELIEF GRAPH INTELLIGENCE (PROBABILISTIC VARIANT):**
Your belief graph contains quantified confidence distributions from aggregated historical evidence.

**Available Information:**
- Confidence levels for each color/rank possibility
- Information-theoretic value of potential actions
- Expected risk/reward calculations for different choices
- Uncertainty reduction potentials

**Strategic Applications:**
- High-confidence cards (>95%) can be played for critical points
- Medium-confidence cards (70-95%) work well for information gathering
- Clues can target cards with highest uncertainty reduction value
- Risk calculations inform optimal play/discard timing decisions
"""

        if variant == 'theory_of_mind':
            result += """
**BELIEF GRAPH INTELLIGENCE (THEORY OF MIND VARIANT):**
Your belief graph contains extracted coordination patterns and strategic priorities from team behavior analysis.

**Available Information:**
- Team_Focus strategic priorities (SAVE_FIVES, PLAY_ONES, PLAY_ANYTHING distributions)
- Teammate skill assessments and complexity tolerance
- Recent strategic shifts and coordination patterns
- Recursive beliefs about how teammates perceive your capabilities

**Strategic Applications:**
- Team_Focus alignment suggests actions that support current strategic momentum
- Skill assessments guide appropriate clue complexity for each teammate
- Recent shifts indicate timing for strategic pivots or coordination plays
- Recursive awareness helps calibrate action expectations and communication style
"""

        if result:
            return result

        else:
            # Fallback for unknown variants
            return """
**BELIEF GRAPH ADVANTAGE:**
Your belief graph provides structured information about the game state. Use this knowledge to make more informed decisions than players relying only on direct observation.
"""

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

## FIREWORKS PROGRESS
"""

        for color, rank in observation['fireworks'].items():
            game_state += f"- {color}: {rank}/5\n"

        # Discard pile
        if observation['discard_pile']:
            game_state += f"\n## DISCARD PILE\n"
            for card in observation['discard_pile']:
                game_state += f"- {card['color']}{card['rank']}\n"
        else:
            game_state += "\n## DISCARD PILE: empty\n"

        # Other players' hands
        game_state += "\n## OTHER PLAYERS' HANDS\n"
        current_player = observation.get('current_player')
        num_players = observation.get('num_players', len(observation['observed_hands']))

        for offset, hand in enumerate(observation['observed_hands'][1:], 1):
            actual_player = None
            if current_player is not None and num_players:
                actual_player = (current_player + offset) % num_players
            display_player = (actual_player + 1) if actual_player is not None else (offset + 1)
            game_state += f"Player {display_player}: "
            for j, card in enumerate(hand, 1):
                color = card.get('color')
                rank = card.get('rank')
                if color is not None and rank is not None and rank >= 0:
                    display_rank = rank + 1
                    display_card = f"{color}{display_rank}"
                    game_state += f"{display_card} "
                else:
                    game_state += "? "
            game_state += "\n"

        # Your hand
        game_state += "\n## YOUR HAND\n"
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
            game_state += "\n"

        # Legal moves
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

        return game_state

    def format_history_for_llm(self, observation_history: List[Dict], action_history: List[Dict]) -> str:
        """Format the observation and action history for the LLM."""
        if not observation_history:
            return "## GAME HISTORY: First turn\n"

        history_text = "## GAME HISTORY\n"

        for i, (obs, action) in enumerate(zip(observation_history, action_history)):
            # Use loop index + 1 for turn numbering to ensure proper incrementing
            turn_num = i + 1
            history_text += f"\n**Turn {turn_num}:**\n"

            if action['action_type'] == 'PLAY':
                history_text += f"- PLAYED card {action['card_index']}\n"
            elif action['action_type'] == 'DISCARD':
                history_text += f"- DISCARDED card {action['card_index']}\n"
            elif action['action_type'] == 'REVEAL_COLOR':
                history_text += f"- COLOR hint ({action['color']}) to player {action['target_offset']}\n"
            elif action['action_type'] == 'REVEAL_RANK':
                history_text += f"- RANK hint ({action['rank'] + 1}) to player {action['target_offset']}\n"

        return history_text

    def create_main_prompt(self, observation: Dict[str, Any], history: str, variant: str = 'certainty') -> str:
        """Create the main prompt for the LLM."""
        game_state = self.format_observation_for_llm(observation)

        # Build prompt sections
        prompt_sections = [
            "You are playing Hanabi-Full-CardKnowledge.",
            "",
            history,
            "",
            self.get_rules_and_mechanics_block(),
            ""
        ]

        # Add belief graph data if present
        if 'belief_graph' in observation:
            prompt_sections.extend([
                observation['belief_graph_natural_language'],
                "",
                self.get_belief_graph_usage_hint(variant),
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
        elif variant in ['probabilistic', 'theory_of_mind']:
            probabilistic_nl = self.format_probabilistic_belief_graph_natural_language(belief_graph)
            if variant == 'theory_of_mind':
                # Add ToM-specific layer on top of probabilistic base
                tom_nl = self.format_theory_of_mind_belief_graph_natural_language(belief_graph)
                return probabilistic_nl + "\n\n" + tom_nl
            return probabilistic_nl
        else:
            # Fallback to certainty format if variant not recognized
            return self.format_certainty_belief_graph_natural_language(belief_graph)

    def format_certainty_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert certainty belief graph to natural language with complete detail."""
        nl_description = "## BELIEF GRAPH ANALYSIS (CERTAINTY VARIANT)\n\n"

        gs = belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"

        nl_description += "**MY HAND BELIEFS:**\n"

        # Get current fireworks for playability checking
        game_state = belief_graph.get('GameState', {})
        fireworks = game_state.get('fireworks', self._get_default_fireworks())

        for card_id, beliefs in belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            card_idx = int(card_num) - 1
            colors = beliefs['possible_colors']
            ranks = beliefs['possible_ranks']

            nl_description += f"- Card {card_num} (card_index {card_idx}):\n"

            # Format possible colors
            if colors:
                color_str = ", ".join([c.upper() for c in sorted(colors)])
                nl_description += f"  * Possible colors: {color_str}\n"
            else:
                nl_description += f"  * Possible colors: Unknown\n"

            # Format possible ranks
            if ranks:
                rank_str = ", ".join([str(r) for r in sorted(ranks)])
                nl_description += f"  * Possible ranks: {rank_str}\n"
            else:
                nl_description += f"  * Possible ranks: Unknown\n"

            # Certainty assessment
            if len(colors) == 1 and len(ranks) == 1:
                color = colors[0]
                rank = ranks[0]
                color_key = color[0].upper() if len(color) > 0 else color.upper()
                current_firework = fireworks.get(color_key, 0)

                if rank == current_firework + 1:
                    nl_description += f"  * Status: Certain - {color.upper()} {rank} (PLAYABLE NOW)\n"
                elif rank <= current_firework:
                    nl_description += f"  * Status: Certain - {color.upper()} {rank} (already played)\n"
                else:
                    nl_description += f"  * Status: Certain - {color.upper()} {rank} (not yet playable)\n"
            elif len(ranks) == 1 and ranks[0] == 1:
                possible_plays = []
                for c in colors:
                    c_key = c[0].upper() if len(c) > 0 else c.upper()
                    if fireworks.get(c_key, 0) == 0:
                        possible_plays.append(c.upper())

                if possible_plays:
                    nl_description += f"  * Status: Rank 1 - can start {'/'.join(possible_plays)}\n"
                else:
                    nl_description += f"  * Status: Rank 1 - all colors already started\n"
            elif len(colors) == 1:
                color = colors[0]
                rank_str = '/'.join(map(str, sorted(ranks)))
                nl_description += f"  * Status: {color.upper()} color, ranks: {rank_str}\n"
            elif len(ranks) == 1:
                rank = ranks[0]
                color_str = ', '.join([c.upper() for c in sorted(colors)])
                nl_description += f"  * Status: Rank {rank}, colors: {color_str}\n"
            else:
                nl_description += f"  * Status: {len(colors)} colors, {len(ranks)} ranks possible\n"

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

                nl_description += f"  - Card {card_num}: I see {actual}\n"

                # What the teammate knows
                if colors and ranks:
                    color_str = ", ".join([c.upper() for c in sorted(colors)])
                    rank_str = ", ".join([str(r) for r in sorted(ranks)])
                    nl_description += f"    * They believe colors: {color_str}\n"
                    nl_description += f"    * They believe ranks: {rank_str}\n"

                    # Their certainty level
                    if len(colors) == 1 and len(ranks) == 1:
                        nl_description += f"    * Their knowledge: Certain ({colors[0].upper()} {ranks[0]})\n"
                    elif len(colors) == 1:
                        nl_description += f"    * Their knowledge: Color known ({colors[0].upper()})\n"
                    elif len(ranks) == 1:
                        nl_description += f"    * Their knowledge: Rank known ({ranks[0]})\n"
                    else:
                        nl_description += f"    * Their knowledge: Uncertain ({len(colors)} colors, {len(ranks)} ranks)\n"
                else:
                    nl_description += f"    * Their knowledge: No belief data\n"

        return nl_description

    def format_probabilistic_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert probabilistic belief graph to natural language with complete detail."""
        nl_description = "## BELIEF GRAPH ANALYSIS (PROBABILISTIC VARIANT)\n\n"

        gs = belief_graph['GameState']
        nl_description += f"**Game State:** {gs['clues']} clues, {gs['life']} lives, {gs['deck_size']} cards in deck\n\n"

        nl_description += "**MY HAND BELIEFS:**\n"

        # Get current fireworks for playability checking
        game_state = belief_graph.get('GameState', {})
        fireworks = game_state.get('fireworks', self._get_default_fireworks())

        for card_id, beliefs in belief_graph['My_Hand_Beliefs'].items():
            card_num = card_id.split('Card')[1]
            card_idx = int(card_num) - 1
            color_dist = beliefs['color_distribution']
            rank_dist = beliefs['rank_distribution']

            # Find highest probability color and rank
            max_color = max(color_dist, key=color_dist.get)
            max_color_prob = color_dist[max_color]
            max_rank = max(rank_dist, key=rank_dist.get)
            max_rank_prob = rank_dist[max_rank]

            # Format color distribution
            color_items = sorted(color_dist.items(), key=lambda x: x[1], reverse=True)
            color_str = ", ".join([f"{c.upper()}({p*100:.0f}%)" for c, p in color_items])

            # Format rank distribution
            rank_items = sorted(rank_dist.items(), key=lambda x: x[1], reverse=True)
            rank_str = ", ".join([f"{r}({p*100:.0f}%)" for r, p in rank_items])

            nl_description += f"- Card {card_num} (card_index {card_idx}):\n"
            nl_description += f"  * Colors: {color_str}\n"
            nl_description += f"  * Ranks: {rank_str}\n"

            # Confidence assessment
            if max_color_prob == 1.0 and max_rank_prob == 1.0:
                nl_description += f"  * Status: Certain - {max_color.upper()} {max_rank}\n"
            elif max_color_prob >= 0.8 and max_rank_prob >= 0.8:
                nl_description += f"  * Status: {max_color_prob*100:.0f}% color, {max_rank_prob*100:.0f}% rank - High confidence\n"
            elif max_color_prob >= 0.8:
                nl_description += f"  * Status: {max_color_prob*100:.0f}% color known ({max_color.upper()}), rank uncertain\n"
            elif max_rank_prob >= 0.8:
                nl_description += f"  * Status: {max_rank_prob*100:.0f}% rank known ({max_rank}), color uncertain\n"
            else:
                nl_description += f"  * Status: {max_color_prob*100:.0f}% color, {max_rank_prob*100:.0f}% rank - Uncertain\n"

        nl_description += "\n**TEAMMATE KNOWLEDGE MODEL:**\n"
        for player_hand, hand_data in belief_graph['Teammate_Hand_Beliefs'].items():
            player_num = player_hand.split('P')[1].split('_')[0]
            nl_description += f"\nPlayer {player_num}'s cards:\n"

            for card_id, card_data in hand_data.items():
                card_num = card_id.split('Card')[1]
                actual = card_data.get('actual_card_I_see', 'Unknown')

                # Extract player number from the belief key instead of player_num from iteration
                belief_key = None
                for key in card_data.keys():
                    if key.endswith('_belief'):
                        belief_key = key
                        break

                belief = card_data.get(belief_key, {})
                color_dist = belief.get('color_distribution', {})
                rank_dist = belief.get('rank_distribution', {})

                # Find highest probability color and rank
                max_color_prob = max(color_dist.values()) if color_dist else 0
                max_rank_prob = max(rank_dist.values()) if rank_dist else 0

                # Format what player believes
                if color_dist and rank_dist:
                    color_items = sorted(color_dist.items(), key=lambda x: x[1], reverse=True)
                    color_str = ", ".join([f"{c.upper()}({p*100:.0f}%)" for c, p in color_items if p > 0])

                    rank_items = sorted(rank_dist.items(), key=lambda x: x[1], reverse=True)
                    rank_str = ", ".join([f"{r}({p*100:.0f}%)" for r, p in rank_items if p > 0])

                    nl_description += f"  - Card {card_num}: I see {actual}\n"
                    nl_description += f"    * They believe colors: {color_str}\n"
                    nl_description += f"    * They believe ranks: {rank_str}\n"

                    # Confidence assessment with raw percentages
                    max_color_prob = max(color_dist.values()) if color_dist else 0
                    max_rank_prob = max(rank_dist.values()) if rank_dist else 0

                    if max_color_prob == 1.0 and max_rank_prob == 1.0:
                        # Find the actual color and rank for certain case
                        certain_color = max(color_dist, key=color_dist.get)
                        certain_rank = max(rank_dist, key=rank_dist.get)
                        nl_description += f"    * Their knowledge: Certain ({certain_color.upper()} {certain_rank})\n"
                    elif max_color_prob >= 0.8 and max_rank_prob >= 0.8:
                        nl_description += f"    * Their knowledge: {max_color_prob*100:.0f}% color, {max_rank_prob*100:.0f}% rank\n"
                    elif max_color_prob >= 0.8:
                        max_color = max(color_dist, key=color_dist.get)
                        nl_description += f"    * Their knowledge: {max_color_prob*100:.0f}% color known ({max_color.upper()})\n"
                    elif max_rank_prob >= 0.8:
                        max_rank = max(rank_dist, key=rank_dist.get)
                        nl_description += f"    * Their knowledge: {max_rank_prob*100:.0f}% rank known ({max_rank})\n"
                    else:
                        nl_description += f"    * Their knowledge: {max_color_prob*100:.0f}% color, {max_rank_prob*100:.0f}% rank\n"
                else:
                    nl_description += f"  - Card {card_num}: I see {actual}, no belief data\n"

        return nl_description

    def format_theory_of_mind_belief_graph_natural_language(self, belief_graph: Dict[str, Any]) -> str:
        """Convert Theory of Mind belief graph to natural language (ToM layer only)."""
        nl_description = "## THEORY OF MIND ANALYSIS\n\n"

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

            nl_description += f"- {player}: Skill {skill:.1f}, Play aggressiveness {aggr:.1f}, Hint quality {hint_q:.1f}\n"

        # Add recursive ToM information
        if 'Recursive_ToM' in tom:
            nl_description += "\n**RECURSIVE ToM (What others think about ME):**\n"
            for player_key, player_recursive in tom['Recursive_ToM'].items():
                nl_description += f"- {player_key} thinks about ME:\n"
                for me_player, beliefs in player_recursive.items():
                    skill_assessment = beliefs.get('believes_about_my_skill', 0.5)
                    strategy_assessment = beliefs.get('believes_about_my_strategy', 0.5)
                    nl_description += f"  * My skill: {skill_assessment:.1f}, My strategy: {strategy_assessment:.1f}\n"

                    # Add detailed beliefs about my hand
                    hand_beliefs = beliefs.get('believes_about_my_hand', {})
                    if hand_beliefs:
                        nl_description += "  * They believe about my hand:\n"
                        for card_id, card_belief in hand_beliefs.items():
                            if isinstance(card_belief, dict):
                                # Handle different belief formats
                                if 'rank' in card_belief and 'confidence' in card_belief:
                                    rank = card_belief['rank']
                                    conf = card_belief['confidence']
                                    color = card_belief.get('color', 'unknown')
                                    nl_description += f"    - {card_id}: {color.upper()} {rank} ({conf*100:.0f}% confident)\n"
                                elif 'color' in card_belief and 'confidence' in card_belief:
                                    color = card_belief['color']
                                    conf = card_belief['confidence']
                                    rank = card_belief.get('rank', 'unknown')
                                    nl_description += f"    - {card_id}: {color.upper()} {rank} ({conf*100:.0f}% confident)\n"
                                else:
                                    # Generic format
                                    details = ", ".join([f"{k}: {v}" for k, v in card_belief.items()])
                                    nl_description += f"    - {card_id}: {details}\n"

        return nl_description
