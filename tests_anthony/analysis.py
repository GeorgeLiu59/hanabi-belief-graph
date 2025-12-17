"""Analysis helper functions for evaluating agent decisions and reasoning.

This module provides functions to evaluate action safety, analyze belief graph usage,
and assess reasoning quality for the ablation study.
"""

from typing import Dict, List, Any
import re


def evaluate_action_safety(action: Dict[str, Any], scenario: Dict[str, Any]) -> str:
    """
    Determine if action is SAFE, RISKY, or FATAL.

    SAFE: Matches ground truth best action or is a reasonable alternative
    RISKY: Suboptimal but not immediately dangerous
    FATAL: Will lose life token or cause game loss

    Args:
        action: Agent's chosen action (dict with 'type' and 'target')
        scenario: Game scenario from ablation_scenarios.json

    Returns:
        "SAFE", "RISKY", or "FATAL"
    """
    dangerous_actions = scenario.get("dangerous_actions", [])
    ground_truth = scenario["ground_truth_best_action"]

    # Check if this is a dangerous/fatal action
    for dangerous in dangerous_actions:
        if _actions_match(action, dangerous):
            return "FATAL"

    # Check if matches ground truth exactly
    if _actions_match(action, ground_truth):
        return "SAFE"

    # Otherwise it's risky (suboptimal but not fatal)
    return "RISKY"


def _actions_match(action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
    """
    Check if two actions are the same.

    Args:
        action1: First action dict
        action2: Second action dict

    Returns:
        True if actions match
    """
    if action1.get("type") != action2.get("type"):
        return False

    action_type = action1.get("type")

    # For PLAY and DISCARD, check target index
    if action_type in ["PLAY", "DISCARD"]:
        return action1.get("target") == action2.get("target")

    # For REVEAL_COLOR, check color and target_offset
    if action_type == "REVEAL_COLOR":
        return (
            action1.get("color") == action2.get("color")
            and action1.get("target_offset") == action2.get("target_offset")
        )

    # For REVEAL_RANK, check rank and target_offset
    if action_type == "REVEAL_RANK":
        return (
            action1.get("rank") == action2.get("rank")
            and action1.get("target_offset") == action2.get("target_offset")
        )

    return False


def analyze_belief_graph_usage(reasoning: str, context_type: str) -> Dict[str, Any]:
    """
    Analyze whether and how agent used belief graph information.

    Returns metrics about belief usage in reasoning.

    Args:
        reasoning: Agent's reasoning text
        context_type: Type of context provided to agent

    Returns:
        Dict with belief usage analysis:
        - applicable: Whether belief analysis is applicable to this context type
        - mentioned_probabilities: Did reasoning mention probabilistic terms?
        - mentioned_impossible: Did reasoning mention impossible cards?
        - used_numerical_values: Did reasoning use numerical probabilities (e.g., "20%")?
        - probability_count: How many times probabilities mentioned?
        - interpretation_correct: Did agent correctly interpret probabilities?
    """
    # Belief usage only applicable to belief-based contexts
    if context_type not in ["full_belief_graph", "natural_language_belief", "certainty_only"]:
        return {
            "applicable": False,
            "mentioned_probabilities": False,
            "mentioned_impossible": False,
            "used_numerical_values": False,
            "probability_count": 0,
            "interpretation_correct": False
        }

    reasoning_lower = reasoning.lower()

    # Check for probabilistic language
    prob_keywords = ["probability", "chance", "likely", "unlikely", "distribution", "percent"]
    mentioned_probabilities = any(word in reasoning_lower for word in prob_keywords)

    # Check for impossibility reasoning
    impossible_phrases = [
        "cannot be", "impossible", "ruled out", "definitely not",
        "already played", "stack is at", "must be"
    ]
    mentioned_impossible = any(phrase in reasoning_lower for phrase in impossible_phrases)

    # Check for numerical probability values
    numerical_pattern = r'\d+\s*%|0\.\d+|probability\s+of\s+\d+'
    used_numerical_values = bool(re.search(numerical_pattern, reasoning))

    # Count probability mentions
    probability_count = (
        reasoning_lower.count("probability")
        + reasoning_lower.count("chance")
        + len(re.findall(r'\d+%', reasoning))
    )

    # Heuristic for correct interpretation:
    # If agent mentions impossible cards correctly (e.g., "White-1 is impossible because White stack is at 3")
    # Look for reasoning that connects fireworks state to impossibility
    interpretation_correct = (
        mentioned_impossible
        and ("stack" in reasoning_lower or "played" in reasoning_lower)
    )

    return {
        "applicable": True,
        "mentioned_probabilities": mentioned_probabilities,
        "mentioned_impossible": mentioned_impossible,
        "used_numerical_values": used_numerical_values,
        "probability_count": probability_count,
        "interpretation_correct": interpretation_correct
    }


def analyze_reasoning_quality(reasoning: str) -> Dict[str, Any]:
    """
    Analyze the quality of agent's reasoning.

    Args:
        reasoning: Agent's reasoning text

    Returns:
        Dict with reasoning quality metrics
    """
    reasoning_lower = reasoning.lower()
    words = reasoning.split()

    return {
        "length_words": len(words),
        "mentions_fireworks": any(
            word in reasoning_lower for word in ["stack", "firework", "fireworks"]
        ),
        "mentions_discards": any(
            word in reasoning_lower for word in ["discard", "discarded"]
        ),
        "mentions_teammate": any(
            word in reasoning_lower for word in ["teammate", "partner", "player 1", "other player"]
        ),
        "coherence_score": check_logical_coherence(reasoning)
    }


def check_logical_coherence(reasoning: str) -> float:
    """
    Score reasoning logical coherence from 0.0 to 1.0.

    Checks for:
    - Premises stated (because, since, given that)
    - Logical flow (if X then Y, therefore, thus)
    - Conclusion matches premises
    - References to game state

    Args:
        reasoning: Agent's reasoning text

    Returns:
        Float score from 0.0 to 1.0
    """
    score = 0.0
    reasoning_lower = reasoning.lower()

    # Has premises? (worth 30%)
    premise_words = ["because", "since", "given that", "based on", "as"]
    if any(word in reasoning_lower for word in premise_words):
        score += 0.3

    # Has logical connectors? (worth 30%)
    connector_words = ["therefore", "so", "thus", "hence", "then", "this means"]
    if any(word in reasoning_lower for word in connector_words):
        score += 0.3

    # Has conclusion/decision? (worth 20%)
    conclusion_phrases = ["i will", "i should", "best to", "i'll", "decide to", "choose to"]
    if any(phrase in reasoning_lower for phrase in conclusion_phrases):
        score += 0.2

    # References game state? (worth 20%)
    game_state_words = [
        "stack", "firework", "discard", "hand", "card", "token",
        "play", "hint", "clue", "reveal"
    ]
    if any(word in reasoning_lower for word in game_state_words):
        score += 0.2

    return min(score, 1.0)


def check_consistency(action: Dict[str, Any], reasoning: str) -> bool:
    """
    Check if action matches what reasoning suggests.

    Example: Reasoning says "I should discard" but action is PLAY → False

    Args:
        action: Agent's chosen action
        reasoning: Agent's reasoning text

    Returns:
        True if action is consistent with reasoning
    """
    action_type = action.get("type", "").upper()
    reasoning_lower = reasoning.lower()

    # Check if action type is mentioned in reasoning
    action_mentioned = action_type.lower() in reasoning_lower

    # Check for contradictions
    if action_type == "PLAY":
        # If reasoning says "discard" but not "play", that's inconsistent
        if "discard" in reasoning_lower and "play" not in reasoning_lower:
            return False

    if action_type == "DISCARD":
        # If reasoning says "play" but not "discard", that's inconsistent
        if "play" in reasoning_lower and "discard" not in reasoning_lower:
            return False

    if action_type in ["REVEAL_COLOR", "REVEAL_RANK"]:
        # If reasoning says "hint" or "clue" or "reveal", that's consistent
        hint_words = ["hint", "clue", "reveal", "tell", "inform"]
        if any(word in reasoning_lower for word in hint_words):
            return True

    # If action is mentioned, consider it consistent
    return action_mentioned


def compare_reasoning_across_contexts(
    results_by_context: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Compare reasoning quality metrics across different context types.

    Args:
        results_by_context: Dict mapping context_type to list of results

    Returns:
        Comparison dict with aggregated metrics per context
    """
    comparison = {}

    for context_type, results in results_by_context.items():
        if not results:
            continue

        total = len(results)

        # Aggregate metrics
        avg_coherence = sum(r.get("coherence_score", 0) for r in results) / total
        avg_length = sum(r.get("reasoning_length_words", 0) for r in results) / total

        fireworks_mentions = sum(
            1 for r in results if r.get("references_fireworks", False)
        )
        prob_mentions = sum(
            1 for r in results if r.get("referenced_probabilities", False)
        )

        comparison[context_type] = {
            "avg_coherence_score": avg_coherence,
            "avg_reasoning_length": avg_length,
            "fireworks_mention_rate": (fireworks_mentions / total * 100) if total > 0 else 0,
            "probability_mention_rate": (prob_mentions / total * 100) if total > 0 else 0,
        }

    return comparison


def identify_turn_28_bug_evidence(
    action: Dict[str, Any],
    reasoning: str,
    scenario: Dict[str, Any],
    context_type: str
) -> Dict[str, Any]:
    """
    Special analysis for Turn 28 bug scenario.

    Checks if agent fell victim to the bug (playing impossible card with 20% probability).

    Args:
        action: Agent's chosen action
        reasoning: Agent's reasoning text
        scenario: Game scenario (should be turn_28_white_stack_bug)
        context_type: Type of context used

    Returns:
        Dict with bug evidence analysis
    """
    if scenario["scenario_name"] != "turn_28_white_stack_bug":
        return {"applicable": False}

    # The bug manifests as: agent plays card 0 (White-1) when White stack is at 3
    is_bug_action = (
        action.get("type") == "PLAY"
        and action.get("target") == 0
    )

    # Check if reasoning mentions the buggy 20% probability
    mentions_20_percent = "20%" in reasoning or "20 percent" in reasoning
    mentions_white = "white" in reasoning.lower()

    # Check if reasoning correctly identifies impossibility
    correctly_identifies_impossible = (
        "impossible" in reasoning.lower()
        and "white" in reasoning.lower()
        and ("stack" in reasoning.lower() or "played" in reasoning.lower())
    )

    return {
        "applicable": True,
        "fell_victim_to_bug": is_bug_action,
        "context_type": context_type,
        "mentioned_20_percent": mentions_20_percent,
        "mentioned_white": mentions_white,
        "correctly_identified_impossible": correctly_identifies_impossible,
        "conclusion": (
            "Agent fell victim to Turn 28 bug - played impossible card"
            if is_bug_action
            else "Agent avoided the bug"
        )
    }
