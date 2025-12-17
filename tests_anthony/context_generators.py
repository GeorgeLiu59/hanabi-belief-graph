"""Context generators for ablation study.

This module generates different context representations from the same game state
to test whether belief graphs improve agent reasoning compared to alternative formats.
"""

from typing import Dict, List, Any
import json


def generate_full_belief_graph_context(scenario: Dict[str, Any]) -> str:
    """
    Generate complete probabilistic belief graph as JSON.

    This is the CURRENT SYSTEM representation - full probabilistic distributions
    for each card showing probabilities for all possible colors and ranks.

    Args:
        scenario: Game state from ablation_scenarios.json

    Returns:
        JSON string with complete belief graph including color/rank probabilities
    """
    player_hand = scenario["player_hands"]["player_0"]
    fireworks = scenario["fireworks"]
    discard_pile = scenario["discard_pile"]

    belief_graph = {"My_Hand_Beliefs": {}}

    # For each card in player's hand
    for idx, card in enumerate(player_hand):
        card_key = f"card_{idx}"
        actual_color = card["color"]
        actual_rank = card["rank"]
        revealed_color = card.get("revealed_color", False)
        revealed_rank = card.get("revealed_rank", False)

        # Calculate color probabilities
        if revealed_color:
            # If color is revealed, 100% certain
            color_dist = {c: 0.0 for c in ["red", "yellow", "green", "white", "blue"]}
            color_dist[actual_color] = 1.0
        else:
            # Uniform distribution (20% each)
            # NOTE: This is buggy - doesn't account for impossible cards from fireworks
            color_dist = {
                "red": 0.2,
                "yellow": 0.2,
                "green": 0.2,
                "white": 0.2,
                "blue": 0.2
            }

        # Calculate rank probabilities
        if revealed_rank:
            # If rank is revealed, 100% certain
            rank_dist = {str(r): 0.0 for r in range(1, 6)}
            rank_dist[str(actual_rank)] = 1.0
        else:
            # Uniform distribution (20% each)
            # NOTE: This is buggy - doesn't account for impossible cards
            rank_dist = {
                "1": 0.2,
                "2": 0.2,
                "3": 0.2,
                "4": 0.2,
                "5": 0.2
            }

        belief_graph["My_Hand_Beliefs"][card_key] = {
            "color_distribution": color_dist,
            "rank_distribution": rank_dist,
            "confidence": 1.0 if (revealed_color and revealed_rank) else 0.5
        }

    return json.dumps(belief_graph, indent=2)


def generate_certainty_only_context(scenario: Dict[str, Any]) -> str:
    """
    Only include certain knowledge (0% or 100% probabilities).

    This representation filters out uncertainty - only shows what we KNOW
    (cards that are impossible or certain), not probabilistic information.

    Args:
        scenario: Game state from ablation_scenarios.json

    Returns:
        JSON string with only certainties (impossible cards and revealed cards)
    """
    player_hand = scenario["player_hands"]["player_0"]
    fireworks = scenario["fireworks"]
    discard_pile = scenario["discard_pile"]

    certainty_graph = {"My_Hand_Certainties": {}}

    for idx, card in enumerate(player_hand):
        card_key = f"card_{idx}"
        actual_color = card["color"]
        actual_rank = card["rank"]
        revealed_color = card.get("revealed_color", False)
        revealed_rank = card.get("revealed_rank", False)

        impossible_cards = []
        certain_info = {}

        # Determine impossible cards based on fireworks
        for color, stack_height in fireworks.items():
            color_name = _stack_key_to_color_name(color)
            # Cards at or below stack height are impossible
            for rank in range(1, stack_height + 1):
                impossible_cards.append((color_name, rank))

        # Add discarded cards as impossible
        for discarded in discard_pile:
            impossible_cards.append((discarded["color"], discarded["rank"]))

        # Certain information from reveals
        if revealed_color:
            certain_info["color"] = actual_color
        if revealed_rank:
            certain_info["rank"] = actual_rank

        certainty_graph["My_Hand_Certainties"][card_key] = {
            "impossible": impossible_cards,
            "certain": certain_info
        }

    return json.dumps(certainty_graph, indent=2)


def generate_raw_game_state_context(scenario: Dict[str, Any]) -> str:
    """
    Just observable facts, no belief representation.

    This shows what the agent can directly SEE without any belief modeling:
    fireworks state, teammate's visible hand, discard pile, tokens.

    Args:
        scenario: Game state from ablation_scenarios.json

    Returns:
        Natural language description of observable game state
    """
    fireworks = scenario["fireworks"]
    life_tokens = scenario["life_tokens"]
    info_tokens = scenario["info_tokens"]
    teammate_hand = scenario["player_hands"]["player_1"]
    discard_pile = scenario["discard_pile"]
    deck_size = scenario["deck_size"]

    # Build natural language description
    context = f"""GAME STATE (Turn {scenario['turn']}):

Fireworks: Red={fireworks['R']}, Yellow={fireworks['Y']}, Green={fireworks['G']}, White={fireworks['W']}, Blue={fireworks['B']}
Life Tokens: {life_tokens}
Info Tokens: {info_tokens}
Cards Remaining in Deck: {deck_size}

Your Hand: {len(scenario['player_hands']['player_0'])} cards (you cannot see your own cards)

Teammate's Visible Hand:
"""

    for idx, card in enumerate(teammate_hand):
        context += f"  Card {idx}: {card['color'].capitalize()}-{card['rank']}"
        if card.get('revealed_color') or card.get('revealed_rank'):
            reveals = []
            if card.get('revealed_color'):
                reveals.append("color revealed")
            if card.get('revealed_rank'):
                reveals.append("rank revealed")
            context += f" ({', '.join(reveals)})"
        context += "\n"

    context += f"\nDiscard Pile ({len(discard_pile)} cards):\n"
    if discard_pile:
        for card in discard_pile:
            context += f"  {card['color'].capitalize()}-{card['rank']}\n"
    else:
        context += "  (empty)\n"

    return context.strip()


def generate_minimal_context(scenario: Dict[str, Any]) -> str:
    """
    Only current hand size and fireworks - bare minimum.

    This tests whether agents can reason effectively with almost no information,
    forcing them to use their own reasoning rather than relying on context.

    Args:
        scenario: Game state from ablation_scenarios.json

    Returns:
        Minimal string with just fireworks and hand size
    """
    fireworks = scenario["fireworks"]
    hand_size = len(scenario["player_hands"]["player_0"])

    context = f"""Fireworks: R={fireworks['R']}, Y={fireworks['Y']}, G={fireworks['G']}, W={fireworks['W']}, B={fireworks['B']}
Your hand: {hand_size} cards"""

    return context


def generate_natural_language_belief_context(scenario: Dict[str, Any]) -> str:
    """
    Belief graph as prose instead of JSON - same information, different format.

    This tests whether representation format matters more than content.
    Contains the same probabilistic information as full_belief_graph but
    presented as natural language instead of structured JSON.

    Args:
        scenario: Game state from ablation_scenarios.json

    Returns:
        Natural language description of belief probabilities
    """
    player_hand = scenario["player_hands"]["player_0"]
    fireworks = scenario["fireworks"]
    discard_pile = scenario["discard_pile"]

    context = "BELIEF ANALYSIS FOR YOUR HAND:\n\n"

    for idx, card in enumerate(player_hand):
        actual_color = card["color"]
        actual_rank = card["rank"]
        revealed_color = card.get("revealed_color", False)
        revealed_rank = card.get("revealed_rank", False)

        context += f"Card {idx}:\n"

        # Color analysis
        if revealed_color:
            context += f"  Color: You KNOW this is {actual_color} (revealed).\n"
        else:
            context += f"  Color: Unknown. Each color has about 20% probability (red, yellow, green, white, blue all equally likely).\n"

        # Rank analysis
        if revealed_rank:
            context += f"  Rank: You KNOW this is rank {actual_rank} (revealed).\n"
        else:
            context += f"  Rank: Unknown. Each rank has about 20% probability (1, 2, 3, 4, 5 all equally likely).\n"

        # Impossibility analysis based on fireworks
        impossible_from_fireworks = []
        for color, stack_height in fireworks.items():
            color_name = _stack_key_to_color_name(color)
            for rank in range(1, stack_height + 1):
                impossible_from_fireworks.append(f"{color_name}-{rank}")

        # Impossibility from discards
        impossible_from_discards = [
            f"{d['color']}-{d['rank']}" for d in discard_pile
        ]

        all_impossible = impossible_from_fireworks + impossible_from_discards

        if all_impossible:
            context += f"  Impossible cards: This DEFINITELY CANNOT BE: {', '.join(all_impossible[:10])}"
            if len(all_impossible) > 10:
                context += f" (and {len(all_impossible) - 10} more)"
            context += "\n"

        context += "\n"

    return context.strip()


# Helper functions

def _stack_key_to_color_name(stack_key: str) -> str:
    """Convert fireworks stack key (R, Y, G, W, B) to full color name."""
    mapping = {
        "R": "red",
        "Y": "yellow",
        "G": "green",
        "W": "white",
        "B": "blue"
    }
    return mapping.get(stack_key, stack_key.lower())


def generate_context(scenario: Dict[str, Any], context_type: str) -> str:
    """
    Dispatch function to generate appropriate context based on type.

    Args:
        scenario: Game state from ablation_scenarios.json
        context_type: One of: "full_belief_graph", "certainty_only",
                     "raw_game_state", "minimal_context", "natural_language_belief"

    Returns:
        Context string in the appropriate format

    Raises:
        ValueError: If context_type is not recognized
    """
    generators = {
        "full_belief_graph": generate_full_belief_graph_context,
        "certainty_only": generate_certainty_only_context,
        "raw_game_state": generate_raw_game_state_context,
        "minimal_context": generate_minimal_context,
        "natural_language_belief": generate_natural_language_belief_context
    }

    if context_type not in generators:
        raise ValueError(
            f"Unknown context_type: {context_type}. "
            f"Must be one of: {list(generators.keys())}"
        )

    return generators[context_type](scenario)
