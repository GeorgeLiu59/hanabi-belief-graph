"""
Natural language belief tests for early game scenarios.

Tests agent decision-making with NL belief descriptions in low-pressure
early game situations (turns 1-8, minimal fireworks, full resources).
"""

import pytest
from tests.shared.nl_belief_generator import generate_nl_beliefs
from tests.shared.llm_judge import evaluate_reasoning, should_judge
from tests.shared.analysis import evaluate_action_safety, analyze_reasoning_quality
from tests.shared.game_stage_classifier import classify_game_stage
from tests.shared.metrics import AgentDecisionMetrics


@pytest.mark.parametrize("scenario_name", [
    "turn_1_game_start",
    "turn_5_minimal_fireworks"
])
def test_early_nl_beliefs(scenario_name, early_scenarios, agent_decision, metrics_collector):
    """
    Test agent decisions with NL beliefs in early game scenarios.

    Args:
        scenario_name: Name of scenario to test
        early_scenarios: Fixture providing early game scenarios
        agent_decision: Fixture providing agent decision function
        metrics_collector: Fixture for collecting test metrics
    """
    scenario = early_scenarios[scenario_name]

    # Verify classification
    game_stage = classify_game_stage(scenario)
    assert game_stage == "early_game", f"Expected early_game, got {game_stage}"

    # Generate NL beliefs
    nl_beliefs = generate_nl_beliefs(scenario)

    print(f"\n{'='*70}")
    print(f"Testing: {scenario_name}")
    print(f"{'='*70}")
    print(f"\nNL Beliefs Generated:\n{nl_beliefs[:200]}...")  # Print first 200 chars

    # Get agent decision
    action, reasoning = agent_decision(scenario, nl_beliefs)

    print(f"\nAgent Action: {action.get('type')} target={action.get('target')}")
    print(f"\nAgent Reasoning:\n{reasoning[:300]}...")  # Print first 300 chars

    # Evaluate safety
    safety = evaluate_action_safety(action, scenario)

    print(f"\nAction Safety: {safety}")

    # Analyze reasoning quality
    reasoning_quality = analyze_reasoning_quality(reasoning)

    # Selective judging
    matches_gt = (action.get("type") == scenario['ground_truth_best_action'].get("type") and
                  action.get("target") == scenario['ground_truth_best_action'].get("target"))

    judge_eval = None
    if should_judge(safety, matches_gt, scenario_name):
        print(f"\n→ Invoking LLM-as-Judge...")
        judge_eval = evaluate_reasoning(scenario, reasoning, action)
        if judge_eval and judge_eval.get("success"):
            print(f"   Judge Score: {judge_eval['composite_score']:.2f}/1.0")
        else:
            print(f"   Judge failed: {judge_eval.get('error') if judge_eval else 'Unknown error'}")
    else:
        print(f"\n→ Skipping judge (safety={safety}, matches_gt={matches_gt})")

    # Collect metrics
    metrics = AgentDecisionMetrics.from_agent_output(
        scenario=scenario,
        game_stage=game_stage,
        action=action,
        reasoning=reasoning,
        safety=safety,
        reasoning_quality=reasoning_quality,
        judge_evaluation=judge_eval
    )

    metrics_collector.add_result(metrics)

    # Assertions
    assert safety != "FATAL", f"Agent chose FATAL action in early game: {action}"
    print(f"\n✓ Test passed for {scenario_name}")
