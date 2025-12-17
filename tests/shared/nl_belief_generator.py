"""Natural language belief description generator for Hanabi scenarios.

Converts structured game state into human-readable belief descriptions that
mimic natural reasoning about card possibilities.

Key features:
1. Describes each card position with plausible identities
2. Enumerates impossibility constraints (cards on stack cannot be in hand)
3. Provides playability context based on current fireworks
4. Summarizes game state (tokens, deck size, pressure)

Design rationale:
- Natural language interface between belief computation and action selection
- Mimics human player reasoning patterns
- Tests whether LLMs can consume and reason with NL beliefs effectively
"""

from typing import Dict, Any, List, Tuple


# ============================================================================
# Card Possibility Analysis
# ============================================================================

def get_playable_cards(fireworks: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Get list of currently playable cards based on fireworks state.

    Args:
        fireworks: Dict mapping color → stack height

    Returns:
        List of (color, rank) tuples for playable cards

    Example:
        {'R': 2, 'Y': 0, 'G': 1, 'W': 3, 'B': 1}
        → [('R', 3), ('Y', 1), ('G', 2), ('W', 4), ('B', 2)]
    """
    playable = []
    for color, height in fireworks.items():
        if height < 5:  # Stack not complete
            playable.append((color, height + 1))
    return playable


def get_impossible_cards(fireworks: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Get list of impossible cards (already on fireworks stacks).

    Args:
        fireworks: Dict mapping color → stack height

    Returns:
        List of (color, rank) tuples for impossible cards

    Example:
        {'R': 3, 'Y': 2, 'G': 0, 'W': 3, 'B': 1}
        → [('R', 1), ('R', 2), ('R', 3), ('Y', 1), ('Y', 2), ('W', 1), ('W', 2), ('W', 3), ('B', 1)]
    """
    impossible = []
    for color, height in fireworks.items():
        for rank in range(1, height + 1):
            impossible.append((color, rank))
    return impossible


def get_critical_cards(fireworks: Dict[str, int]) -> List[Tuple[str, int]]:
    """
    Get list of critical cards (rank 5s that haven't been played yet).

    Args:
        fireworks: Dict mapping color → stack height

    Returns:
        List of (color, rank) tuples for critical cards
    """
    critical = []
    for color, height in fireworks.items():
        if height < 5:  # 5 not yet played
            critical.append((color, 5))
    return critical


# ============================================================================
# Natural Language Formatting
# ============================================================================

def format_card_name(color: str, rank: int) -> str:
    """Format card as human-readable string."""
    color_names = {
        'R': 'Red',
        'Y': 'Yellow',
        'G': 'Green',
        'W': 'White',
        'B': 'Blue'
    }
    return f"{color_names.get(color, color)}-{rank}"


def describe_card_possibilities(
    card_index: int,
    fireworks: Dict[str, int],
    card_knowledge: Dict[str, Any],
    discard_pile: List[Dict[str, Any]]
) -> str:
    """
    Generate natural language description of card possibilities.

    Args:
        card_index: Position in hand (0-indexed)
        fireworks: Current fireworks state
        card_knowledge: Known information about this card (color/rank hints)
        discard_pile: Cards already discarded

    Returns:
        Multi-line natural language description
    """
    lines = []
    lines.append(f"- Card {card_index + 1} (position {card_index}):")

    # Extract known information
    known_color = card_knowledge.get('color')
    known_rank = card_knowledge.get('rank')

    # Get context
    playable = get_playable_cards(fireworks)
    impossible = get_impossible_cards(fireworks)

    # Describe what's known
    if known_color and known_rank:
        lines.append(f"  * Known to be {format_card_name(known_color, known_rank)}")
        if (known_color, known_rank) in playable:
            lines.append(f"  * PLAYABLE NOW (next card for {known_color} stack)")
        elif (known_color, known_rank) in impossible:
            lines.append(f"  * IMPOSSIBLE (already on stack) - DO NOT PLAY")
        else:
            lines.append(f"  * Not yet playable (stack needs lower ranks first)")
    elif known_color:
        lines.append(f"  * Known color: {known_color}")
        stack_height = fireworks[known_color]
        if stack_height < 5:
            lines.append(f"  * Could be {known_color}-{stack_height + 1} through {known_color}-5")
            lines.append(f"  * Cannot be {known_color}-1 through {known_color}-{stack_height} (already played)")
        else:
            lines.append(f"  * Stack complete - this card is useless")
    elif known_rank:
        lines.append(f"  * Known rank: {known_rank}")
        possible_colors = []
        for color, height in fireworks.items():
            if (color, known_rank) not in impossible:
                possible_colors.append(color)
        lines.append(f"  * Could be any of: {', '.join(possible_colors)}")
    else:
        lines.append(f"  * No hints received - unknown color and rank")
        lines.append(f"  * Could be any card not yet played or discarded")

    # Add impossibility constraints
    if not (known_color and known_rank):
        impossible_str = ", ".join([format_card_name(c, r) for c, r in impossible[:5]])
        if len(impossible) > 5:
            impossible_str += f" (and {len(impossible) - 5} more)"
        lines.append(f"  * Cannot be any cards already on stacks: {impossible_str}")

    return "\n".join(lines)


def describe_game_state(scenario: Dict[str, Any]) -> str:
    """
    Generate natural language summary of game state.

    Args:
        scenario: Full scenario dictionary

    Returns:
        Multi-line game state summary
    """
    lines = []
    lines.append("Game state summary:")

    # Fireworks
    fw = scenario['fireworks']
    fw_str = ", ".join([f"{c}-{h}" for c, h in fw.items()])
    lines.append(f"- Fireworks: {fw_str}")
    total_played = sum(fw.values())
    lines.append(f"  * Total cards played: {total_played}/25")

    # Tokens
    info_tokens = scenario.get('info_tokens', scenario.get('information_tokens', 8))
    lines.append(f"- Information tokens: {info_tokens}/8", end="")
    if info_tokens >= 6:
        lines[-1] += " (plenty available for hints)"
    elif info_tokens >= 3:
        lines[-1] += " (moderate - use carefully)"
    elif info_tokens >= 1:
        lines[-1] += " (scarce - critical shortage)"
    else:
        lines[-1] += " (NONE - cannot give hints)"

    life_tokens = scenario['life_tokens']
    lines.append(f"- Life tokens: {life_tokens}/3", end="")
    if life_tokens == 3:
        lines[-1] += " (full health)"
    elif life_tokens == 2:
        lines[-1] += " (one mistake made)"
    elif life_tokens == 1:
        lines[-1] += " (CRITICAL - one more mistake ends game)"
    else:
        lines[-1] += " (GAME OVER)"

    # Deck
    deck_size = scenario.get('deck_size', 50)
    lines.append(f"- Deck remaining: {deck_size} cards")
    if deck_size <= 10:
        lines.append(f"  * ENDGAME COUNTDOWN - only {deck_size} draws left")
    elif deck_size <= 20:
        lines.append(f"  * Late game - deck running low")

    # Turn
    turn = scenario.get('turn', 1)
    lines.append(f"- Turn number: {turn}")

    return "\n".join(lines)


# ============================================================================
# Main Generation Function
# ============================================================================

def generate_nl_beliefs(scenario: Dict[str, Any]) -> str:
    """
    Generate natural language belief description for scenario.

    This is the main entry point for converting structured game state
    into human-readable belief descriptions.

    Args:
        scenario: Game scenario dictionary containing:
            - player_hands: Dict with player_0, player_1 hands
            - fireworks: Dict mapping color → stack height
            - info_tokens: Number of information tokens remaining
            - life_tokens: Number of life tokens remaining
            - deck_size: Cards remaining in deck
            - discard_pile: List of discarded cards
            - turn: Current turn number (optional)

    Returns:
        Multi-line natural language belief description string

    Example output:
        ```
        Your beliefs about your hand:
        - Card 1 (position 0):
          * Known color: Red
          * Could be Red-4 or Red-5
          * Cannot be Red-1 through Red-3 (already played)

        - Card 2 (position 1):
          * No hints received - unknown color and rank
          * Could be any card not yet played or discarded
          * Cannot be any cards already on stacks: Red-1, Red-2, Red-3, White-1, White-2

        Game state summary:
        - Fireworks: R-3, Y-2, G-0, W-2, B-1
        - Total cards played: 8/25
        - Information tokens: 7/8 (plenty available for hints)
        - Life tokens: 3/3 (full health)
        - Deck remaining: 42 cards
        - Turn number: 5
        ```
    """
    lines = []

    # Header
    lines.append("Your beliefs about your hand:")
    lines.append("")

    # Describe each card in agent's hand
    agent_hand = scenario['player_hands']['player_0']
    fireworks = scenario['fireworks']
    discard_pile = scenario.get('discard_pile', [])

    # Get card knowledge if available
    card_knowledge_list = scenario.get('card_knowledge', [])
    if not card_knowledge_list:
        # Generate default unknown knowledge
        card_knowledge_list = [{'color': None, 'rank': None} for _ in agent_hand]

    for i, card in enumerate(agent_hand):
        card_knowledge = card_knowledge_list[i] if i < len(card_knowledge_list) else {'color': None, 'rank': None}
        description = describe_card_possibilities(i, fireworks, card_knowledge, discard_pile)
        lines.append(description)
        lines.append("")

    # Game state summary
    lines.append(describe_game_state(scenario))

    return "\n".join(lines)


# ============================================================================
# Utility Functions
# ============================================================================

def generate_nl_beliefs_with_explicit_impossibility(
    scenario: Dict[str, Any],
    highlight_impossible: bool = True
) -> str:
    """
    Enhanced version that explicitly highlights impossible cards.

    This is useful for testing scenarios like Turn 28 bug where
    we want to emphasize impossibility constraints.

    Args:
        scenario: Game scenario dictionary
        highlight_impossible: Whether to add explicit impossibility section

    Returns:
        Natural language description with impossibility emphasis
    """
    base_description = generate_nl_beliefs(scenario)

    if not highlight_impossible:
        return base_description

    # Add explicit impossibility section
    impossible = get_impossible_cards(scenario['fireworks'])

    impossibility_section = [
        "",
        "CRITICAL CONSTRAINT - Cards that CANNOT be in your hand:",
        "The following cards are already on the fireworks stacks and therefore",
        "have 0% probability of being in your hand:"
    ]

    # Group by color
    by_color = {}
    for color, rank in impossible:
        if color not in by_color:
            by_color[color] = []
        by_color[color].append(rank)

    for color, ranks in sorted(by_color.items()):
        color_name = {'R': 'Red', 'Y': 'Yellow', 'G': 'Green', 'W': 'White', 'B': 'Blue'}.get(color, color)
        rank_str = ", ".join([f"{color_name}-{r}" for r in sorted(ranks)])
        impossibility_section.append(f"- {rank_str}")

    return base_description + "\n" + "\n".join(impossibility_section)


# TODO (for Yash/Anthony):
# 1. Add support for card knowledge from hints (integrate hint history)
# 2. Consider discard pile composition (track which cards are gone)
# 3. Add probability estimates if available from belief graph
# 4. Test with Turn 28 bug scenario to ensure impossibility is clear
# 5. Optimize for different game stages (early vs late game descriptions)
