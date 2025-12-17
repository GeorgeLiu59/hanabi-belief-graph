"""
Natural language belief tests for late game scenarios.

Tests agent decision-making with NL belief descriptions in high-pressure
late game situations (turn ≥20, or critical resources, or near-endgame).

Includes the critical Turn 28 bug test.
"""

import pytest
from tests.shared.nl_belief_generator import generate_nl_beliefs, generate_nl_beliefs_with_explicit_impossibility
from tests.shared.llm_judge import evaluate_reasoning, should_judge
from tests.shared.analysis import (
    evaluate_action_safety,
    analyze_reasoning_quality,
    identify_turn_28_bug_evidence
)
from tests.shared.game_stage_classifier import classify_game_stage
from tests.shared.metrics import AgentDecisionMetrics


@pytest.mark.parametrize("scenario_name", [
    "turn_42_late_game",
    "info_tokens_depleted",
    "last_life_token"
])
def test_late_nl_beliefs(scenario_name, end_scenarios, agent_decision, metrics_collector):
    """
    Test agent decisions with NL beliefs in late game scenarios.

    Args:
        scenario_name: Name of scenario to test
        end_scenarios: Fixture providing late game scenarios
        agent_decision: Fixture providing agent decision function
        metrics_collector: Fixture for collecting test metrics
    """
    scenario = end_scenarios[scenario_name]

    # Verify classification
    game_stage = classify_game_stage(scenario)
    assert game_stage == "late_game", f"Expected late_game, got {game_stage}"

    # Generate NL beliefs
    nl_beliefs = generate_nl_beliefs(scenario)

    print(f"\n{'='*70}")
    print(f"Testing: {scenario_name}")
    print(f"{'='*70}")
    print(f"\nNL Beliefs Generated:\n{nl_beliefs[:200]}...")

    # Get agent decision
    action, reasoning = agent_decision(scenario, nl_beliefs)

    print(f"\nAgent Action: {action.get('type')} target={action.get('target')}")
    print(f"\nAgent Reasoning:\n{reasoning[:300]}...")

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
            print(f"   FA: {judge_eval['factual_accuracy']}, IR: {judge_eval['impossibility_recognition']}, "
                  f"LS: {judge_eval['logical_soundness']}, AA: {judge_eval['action_alignment']}")
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

    # Assertions - late game should be very cautious
    if scenario['life_tokens'] == 1:
        assert safety != "FATAL", f"Agent chose FATAL action with last life token: {action}"

    print(f"\n✓ Test passed for {scenario_name}")


def test_turn_28_bug_scenario(turn_28_bug_scenario, agent_decision, metrics_collector):
    """
    CRITICAL TEST: Turn 28 bug scenario.

    Tests if agent correctly identifies that White-1/2/3 are impossible
    (White stack at 3) and avoids playing card 0.

    This is the primary bug test for the belief system correctness.
    """
    scenario = turn_28_bug_scenario

    # Verify classification
    game_stage = classify_game_stage(scenario)
    assert game_stage == "late_game", f"Expected late_game, got {game_stage}"
    assert scenario.get("is_bug_scenario") == True, "Bug scenario flag should be set"

    # Generate NL beliefs WITH EXPLICIT IMPOSSIBILITY EMPHASIS
    nl_beliefs = generate_nl_beliefs_with_explicit_impossibility(
        scenario,
        highlight_impossible=True
    )

    print(f"\n{'='*70}")
    print(f"CRITICAL BUG TEST: {scenario['scenario_name']}")
    print(f"{'='*70}")
    print(f"\nWhite Stack at 3 → White-1/2/3 are IMPOSSIBLE")
    print(f"\nNL Beliefs Generated:\n{nl_beliefs[:300]}...")

    # Get agent decision
    action, reasoning = agent_decision(scenario, nl_beliefs)

    print(f"\nAgent Action: {action.get('type')} target={action.get('target')}")
    print(f"\nAgent Reasoning:\n{reasoning[:500]}...")

    # Evaluate safety
    safety = evaluate_action_safety(action, scenario)

    print(f"\nAction Safety: {safety}")

    # Analyze bug-specific evidence
    bug_evidence = identify_turn_28_bug_evidence(action, reasoning, scenario)

    print(f"\nBug Analysis:")
    print(f"  Fell victim to bug: {bug_evidence['fell_victim_to_bug']}")
    print(f"  Mentioned impossible: {bug_evidence['mentioned_impossible']}")
    print(f"  Correctly identified impossible: {bug_evidence['correctly_identified_impossible']}")
    print(f"  Conclusion: {bug_evidence['conclusion']}")

    # Analyze reasoning quality
    reasoning_quality = analyze_reasoning_quality(reasoning)

    # ALWAYS JUDGE Turn 28 bug scenario
    print(f"\n→ Invoking LLM-as-Judge (ALWAYS for bug scenario)...")
    judge_eval = evaluate_reasoning(scenario, reasoning, action)

    if judge_eval and judge_eval.get("success"):
        print(f"   Judge Score: {judge_eval['composite_score']:.2f}/1.0")
        print(f"   Factual Accuracy: {judge_eval['factual_accuracy']}")
        print(f"   Impossibility Recognition: {judge_eval['impossibility_recognition']}")
        print(f"   Logical Soundness: {judge_eval['logical_soundness']}")
        print(f"   Action Alignment: {judge_eval['action_alignment']}")
        print(f"\n   Explanation: {judge_eval['explanation']}")
    else:
        print(f"   Judge failed: {judge_eval.get('error') if judge_eval else 'Unknown error'}")

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

    # CRITICAL ASSERTION
    if bug_evidence['fell_victim_to_bug']:
        print(f"\n❌ BUG TEST FAILED: Agent played impossible card (White-1)")
        print(f"   This indicates belief system incorrectly assigned non-zero probability")
        print(f"   to cards already on stacks.")
    else:
        print(f"\n✅ BUG TEST PASSED: Agent avoided playing impossible card")

    # Assert that agent did NOT fall victim to bug
    assert not bug_evidence['fell_victim_to_bug'], \
        "Turn 28 BUG: Agent played card 0 (White-1) when White stack at 3 - impossible card!"

    # Assert that agent recognized impossibility (for high-quality reasoning)
    if judge_eval and judge_eval.get("success"):
        assert judge_eval['impossibility_recognition'] != 'INCORRECT', \
            "Agent failed to recognize impossible cards in Turn 28 bug scenario"

    print(f"\n✓ Turn 28 bug test PASSED")
