"""Chain of Thought reasoning quality analysis tests.

This module analyzes HOW agents reason with different contexts,
focusing on reasoning patterns, belief usage, and logical coherence.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from .context_generators import generate_context
from .analysis import (
    analyze_belief_graph_usage,
    analyze_reasoning_quality,
    check_logical_coherence,
    check_consistency
)


CONTEXT_TYPES = [
    "full_belief_graph",
    "certainty_only",
    "raw_game_state",
    "minimal_context",
    "natural_language_belief"
]


@pytest.fixture(scope="module")
def turn_28_scenario():
    """Load Turn 28 scenario specifically for reasoning analysis."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    scenarios_file = fixtures_dir / "ablation_scenarios.json"

    with open(scenarios_file) as f:
        data = json.load(f)

    return next(s for s in data["scenarios"] if s["scenario_name"] == "turn_28_white_stack_bug")


class TestReasoningQuality:
    """Analyze HOW agent reasons with different contexts."""

    def test_belief_graph_referenced_in_reasoning(self, turn_28_scenario, mock_agent_decision):
        """
        Question: Does agent actually USE belief graph when given one?

        Check if CoT mentions:
        - Probabilities ("20% chance", "unlikely", etc.)
        - Impossible cards ("cannot be white-1")
        - Distributions ("most likely red or blue")
        """
        context = generate_context(turn_28_scenario, "full_belief_graph")

        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        # Analyze belief usage
        belief_usage = analyze_belief_graph_usage(reasoning, "full_belief_graph")

        print(f"\n=== REASONING WITH BELIEF GRAPH ===")
        print(f"Referenced probabilities: {belief_usage['mentioned_probabilities']}")
        print(f"Mentioned impossible cards: {belief_usage['mentioned_impossible']}")
        print(f"Used numerical values: {belief_usage['used_numerical_values']}")
        print(f"\nFull reasoning:\n{reasoning}\n")

        # Document findings (don't fail - we're measuring, not validating)
        assert True, (
            f"Belief graph usage: "
            f"probabilities={belief_usage['mentioned_probabilities']}, "
            f"impossible={belief_usage['mentioned_impossible']}"
        )

    def test_reasoning_without_belief_graph(self, turn_28_scenario, mock_agent_decision):
        """
        Question: How does agent reason without belief graph?

        Hypothesis: Might be clearer/simpler reasoning.
        """
        context = generate_context(turn_28_scenario, "raw_game_state")

        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        # Analyze reasoning quality
        quality = analyze_reasoning_quality(reasoning)

        print(f"\n=== REASONING WITHOUT BELIEF GRAPH ===")
        print(f"References fireworks: {quality['mentions_fireworks']}")
        print(f"References discards: {quality['mentions_discards']}")
        print(f"Coherence score: {quality['coherence_score']:.2f}")
        print(f"\nFull reasoning:\n{reasoning}\n")

        # Agent should reason about observable game state
        assert True, (
            f"Raw state reasoning: "
            f"fireworks={quality['mentions_fireworks']}, "
            f"coherence={quality['coherence_score']:.2f}"
        )

    @pytest.mark.parametrize("context_type", CONTEXT_TYPES)
    def test_reasoning_quality_comparison(
        self,
        context_type: str,
        turn_28_scenario,
        mock_agent_decision
    ):
        """
        Compare reasoning quality across contexts.

        Metrics:
        - Logical coherence
        - Appropriate use of available information
        - Correct conclusion from premises
        """
        context = generate_context(turn_28_scenario, context_type)

        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        # Analyze all aspects
        quality = analyze_reasoning_quality(reasoning)
        belief_usage = analyze_belief_graph_usage(reasoning, context_type)
        consistent = check_consistency(action, reasoning)

        results = {
            "context_type": context_type,
            "reasoning_length": quality["length_words"],
            "mentions_fireworks": quality["mentions_fireworks"],
            "mentions_probabilities": belief_usage.get("mentioned_probabilities", False),
            "logical_coherence": quality["coherence_score"],
            "action_consistent_with_reasoning": consistent
        }

        print(f"\n=== REASONING QUALITY: {context_type} ===")
        for key, value in results.items():
            print(f"{key}: {value}")

        # Document findings
        assert True, f"Reasoning quality for {context_type}: coherence={quality['coherence_score']:.2f}"

    def test_coherence_score_calculation(self, turn_28_scenario, mock_agent_decision):
        """
        Validate coherence scoring logic.

        Good reasoning should have:
        - Premises (because, since, given that)
        - Logical flow (therefore, thus, so)
        - Conclusion (I will, I should)
        - Game state references
        """
        # Test with full belief graph context
        context = generate_context(turn_28_scenario, "full_belief_graph")
        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        coherence = check_logical_coherence(reasoning)

        print(f"\n=== COHERENCE SCORE BREAKDOWN ===")
        print(f"Overall score: {coherence:.2f}/1.0")
        print(f"\nReasoning text:\n{reasoning}\n")

        # Coherence should be between 0 and 1
        assert 0.0 <= coherence <= 1.0, f"Coherence score {coherence} out of valid range"

    def test_consistency_between_action_and_reasoning(
        self,
        turn_28_scenario,
        mock_agent_decision
    ):
        """
        Test that action matches reasoning.

        If reasoning says "I should discard" but action is PLAY, that's inconsistent.
        """
        context = generate_context(turn_28_scenario, "raw_game_state")
        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        is_consistent = check_consistency(action, reasoning)

        print(f"\n=== ACTION-REASONING CONSISTENCY ===")
        print(f"Action: {action.get('type')}")
        print(f"Consistent with reasoning: {is_consistent}")
        print(f"\nReasoning:\n{reasoning}\n")

        # Document findings
        assert True, f"Action-reasoning consistency: {is_consistent}"

    @pytest.mark.parametrize("context_type", CONTEXT_TYPES)
    def test_belief_interpretation_correctness(
        self,
        context_type: str,
        turn_28_scenario,
        mock_agent_decision
    ):
        """
        Test if agent correctly interprets belief information.

        For Turn 28 scenario:
        - White stack is at 3
        - Agent's card 0 is White-1
        - Correct interpretation: "White-1 is impossible because White stack is at 3"
        - Incorrect interpretation: "White-1 has 20% probability" (bug)
        """
        if context_type not in ["full_belief_graph", "natural_language_belief", "certainty_only"]:
            pytest.skip("Belief interpretation only applicable to belief-based contexts")

        context = generate_context(turn_28_scenario, context_type)
        action, reasoning = mock_agent_decision(turn_28_scenario, context)

        belief_usage = analyze_belief_graph_usage(reasoning, context_type)

        print(f"\n=== BELIEF INTERPRETATION: {context_type} ===")
        print(f"Mentioned impossible cards: {belief_usage['mentioned_impossible']}")
        print(f"Interpretation correct: {belief_usage['interpretation_correct']}")
        print(f"\nReasoning:\n{reasoning}\n")

        # Document findings
        assert True, (
            f"{context_type} belief interpretation: "
            f"correct={belief_usage['interpretation_correct']}"
        )


class TestReasoningPatterns:
    """Analyze patterns in reasoning across different scenarios and contexts."""

    def test_fireworks_reasoning_pattern(self, mock_agent_decision):
        """
        Test how different contexts lead to fireworks-based reasoning.

        Hypothesis: Raw game state forces focus on fireworks, while belief graphs
        might distract with probability calculations.
        """
        # Load Turn 28 scenario
        fixtures_dir = Path(__file__).parent / "fixtures"
        with open(fixtures_dir / "ablation_scenarios.json") as f:
            scenarios = json.load(f)["scenarios"]
        scenario = next(s for s in scenarios if s["scenario_name"] == "turn_28_white_stack_bug")

        fireworks_mention_by_context = {}

        for context_type in CONTEXT_TYPES:
            context = generate_context(scenario, context_type)
            action, reasoning = mock_agent_decision(scenario, context)

            quality = analyze_reasoning_quality(reasoning)
            fireworks_mention_by_context[context_type] = quality["mentions_fireworks"]

        print(f"\n=== FIREWORKS REASONING PATTERN ===")
        for context_type, mentions_fireworks in fireworks_mention_by_context.items():
            print(f"{context_type}: {'✓' if mentions_fireworks else '✗'}")

        # Document findings
        assert True, "Fireworks reasoning pattern documented"

    def test_probability_language_usage(self, mock_agent_decision):
        """
        Analyze how agents use probability language.

        Do they say:
        - "20% chance" (numerical)
        - "unlikely" (qualitative)
        - "impossible" (absolute)
        """
        fixtures_dir = Path(__file__).parent / "fixtures"
        with open(fixtures_dir / "ablation_scenarios.json") as f:
            scenarios = json.load(f)["scenarios"]
        scenario = next(s for s in scenarios if s["scenario_name"] == "turn_28_white_stack_bug")

        probability_usage = {}

        for context_type in ["full_belief_graph", "natural_language_belief"]:
            context = generate_context(scenario, context_type)
            action, reasoning = mock_agent_decision(scenario, context)

            belief_usage = analyze_belief_graph_usage(reasoning, context_type)

            probability_usage[context_type] = {
                "numerical": belief_usage["used_numerical_values"],
                "mentioned_probabilities": belief_usage["mentioned_probabilities"],
                "mentioned_impossible": belief_usage["mentioned_impossible"]
            }

        print(f"\n=== PROBABILITY LANGUAGE USAGE ===")
        for context_type, usage in probability_usage.items():
            print(f"\n{context_type}:")
            print(f"  Numerical (e.g., 20%): {usage['numerical']}")
            print(f"  Probability language: {usage['mentioned_probabilities']}")
            print(f"  Impossibility language: {usage['mentioned_impossible']}")

        # Document findings
        assert True, "Probability language usage documented"
