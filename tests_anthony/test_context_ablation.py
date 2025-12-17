"""Core ablation tests comparing agent performance across context types.

This module runs the main ablation study: testing each scenario with all 5 context
representations and capturing agent decisions, reasoning, and performance metrics.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from .context_generators import generate_context
from .analysis import (
    evaluate_action_safety,
    analyze_belief_graph_usage,
    analyze_reasoning_quality,
    identify_turn_28_bug_evidence
)
from .metrics import AgentDecisionMetrics


# Context types to test
CONTEXT_TYPES = [
    "full_belief_graph",
    "certainty_only",
    "raw_game_state",
    "minimal_context",
    "natural_language_belief"
]


@pytest.fixture(scope="module")
def ablation_scenarios():
    """Load pre-defined identical scenarios for ablation testing."""
    fixtures_dir = Path(__file__).parent / "fixtures"
    scenarios_file = fixtures_dir / "ablation_scenarios.json"

    with open(scenarios_file) as f:
        data = json.load(f)

    return data["scenarios"]


@pytest.mark.parametrize("context_type", CONTEXT_TYPES)
class TestContextAblation:
    """Compare agent decisions across different context representations."""

    def test_turn_1_game_start(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test Turn 1 scenario - game start with no information."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "turn_1_game_start")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def test_turn_5_minimal_fireworks(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test Turn 5 scenario - some fireworks started."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "turn_5_minimal_fireworks")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def test_turn_10_medium_complexity(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test Turn 10 scenario - mid-game with multiple stacks building."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "turn_10_medium_complexity")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def test_turn_28_white_stack_bug(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """
        CRITICAL TEST: Turn 28 bug scenario.

        Expected insights:
        - Does full belief graph prevent bad action? (probably NO due to bug)
        - Does raw game state lead to better decision? (possibly YES)
        - Does agent use belief information at all?

        This is the most important test in the ablation study.
        """
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "turn_28_white_stack_bug")

        # Run the standard test
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

        # Additional Turn 28 bug-specific analysis
        context = generate_context(scenario, context_type)
        action, reasoning = mock_agent_decision(scenario, context)

        bug_evidence = identify_turn_28_bug_evidence(action, reasoning, scenario, context_type)

        if bug_evidence["applicable"]:
            # Log bug-specific findings
            print(f"\n=== TURN 28 BUG ANALYSIS ({context_type}) ===")
            print(f"Fell victim to bug: {bug_evidence['fell_victim_to_bug']}")
            print(f"Conclusion: {bug_evidence['conclusion']}\n")

    def test_turn_42_late_game(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test Turn 42 scenario - late game with high stakes."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "turn_42_late_game")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def test_info_tokens_depleted(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test info tokens depleted scenario - must play or discard."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "info_tokens_depleted")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def test_last_life_token(
        self,
        context_type: str,
        ablation_scenarios: list,
        mock_agent_decision,
        metrics_collector
    ):
        """Test last life token scenario - extremely high risk."""
        scenario = next(s for s in ablation_scenarios if s["scenario_name"] == "last_life_token")
        self._run_scenario_test(scenario, context_type, mock_agent_decision, metrics_collector)

    def _run_scenario_test(
        self,
        scenario: Dict[str, Any],
        context_type: str,
        agent_decision_func,
        metrics_collector
    ):
        """
        Core test logic shared across all scenarios.

        Args:
            scenario: Game scenario from ablation_scenarios.json
            context_type: Type of context to generate
            agent_decision_func: Function that returns (action, reasoning)
            metrics_collector: Shared metrics collector fixture
        """
        # Generate context based on type
        context = generate_context(scenario, context_type)

        # Get agent decision + reasoning
        action, reasoning = agent_decision_func(scenario, context)

        # Evaluate decision quality
        safety = evaluate_action_safety(action, scenario)

        # Analyze belief graph usage
        belief_usage = analyze_belief_graph_usage(reasoning, context_type)

        # Analyze reasoning quality
        reasoning_quality = analyze_reasoning_quality(reasoning)

        # Create metrics object
        metrics = AgentDecisionMetrics.from_agent_output(
            scenario=scenario,
            context_type=context_type,
            action=action,
            reasoning=reasoning,
            safety=safety,
            belief_usage=belief_usage,
            reasoning_quality=reasoning_quality
        )

        # Add to collector
        metrics_collector.add_result(metrics)

        # Log results for visibility during test run
        print(f"\n=== {scenario['scenario_name']} | {context_type} ===")
        print(f"Action: {action.get('type')} (target {action.get('target', 'N/A')})")
        print(f"Safety: {safety}")
        print(f"Matches ground truth: {metrics.matches_ground_truth}")
        print(f"Reasoning length: {reasoning_quality['length_words']} words")
        print(f"Coherence score: {reasoning_quality['coherence_score']:.2f}")

        # Assert and document
        # NOTE: We DON'T fail tests based on agent performance - we're measuring, not validating
        # The assertions are just for documentation purposes
        self._assert_and_document(
            actual=action,
            expected=scenario["ground_truth_best_action"],
            context=context_type,
            safety=safety,
            scenario_name=scenario["scenario_name"]
        )

    def _assert_and_document(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
        context: str,
        safety: str,
        scenario_name: str
    ):
        """
        Document test results without failing tests.

        The ablation study is about MEASURING performance, not VALIDATING correctness.
        We want all tests to complete so we can compare results.
        """
        matches = (
            actual.get("type") == expected.get("type")
            and actual.get("target") == expected.get("target")
        )

        # Always pass - we're collecting data, not enforcing correctness
        assert True, (
            f"{context} context on {scenario_name}: "
            f"Action={actual.get('type')} (target {actual.get('target', 'N/A')}), "
            f"Safety={safety}, "
            f"Matches_GT={matches}"
        )


@pytest.mark.skip(reason="Placeholder for future scenario-specific deep dives")
class TestScenarioDeepDives:
    """
    Optional: Detailed analysis of specific scenarios.

    These tests can be used to investigate interesting scenarios in more detail,
    examining agent reasoning patterns, decision trees, etc.
    """

    def test_turn_28_reasoning_patterns(self, ablation_scenarios, metrics_collector):
        """Analyze reasoning patterns specifically for Turn 28 bug scenario."""
        pass

    def test_late_game_risk_assessment(self, ablation_scenarios, metrics_collector):
        """Analyze how different contexts affect risk assessment in late game."""
        pass
