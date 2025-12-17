"""
pytest fixtures for Hanabi natural language belief ablation study.

Provides reusable test fixtures for:
- Stage-specific scenario loading (early, mid, late game)
- Agent instances with NL belief injection
- Turn 28 bug scenario
- Metrics collection and session-level reporting
"""

import pytest
import sys
import os
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import ablation study components
from tests.shared.metrics import MetricsCollector

# Global metrics collector shared across all tests
_metrics_collector = MetricsCollector()


# ============================================================================
# STAGE-SPECIFIC SCENARIO FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def early_scenarios():
    """
    Load early game scenarios from hanabi_early/scenarios.json.

    Returns:
        Dict[str, dict]: Mapping of scenario_name to scenario dict
    """
    scenarios_path = Path(__file__).parent.parent / "hanabi_early" / "scenarios.json"

    with open(scenarios_path, 'r') as f:
        data = json.load(f)

    # Create dict mapping scenario_name to scenario
    scenarios_dict = {s["scenario_name"]: s for s in data["scenarios"]}
    return scenarios_dict


@pytest.fixture(scope="session")
def mid_scenarios():
    """
    Load mid game scenarios from hanabi_mid/scenarios.json.

    Returns:
        Dict[str, dict]: Mapping of scenario_name to scenario dict
    """
    scenarios_path = Path(__file__).parent.parent / "hanabi_mid" / "scenarios.json"

    with open(scenarios_path, 'r') as f:
        data = json.load(f)

    scenarios_dict = {s["scenario_name"]: s for s in data["scenarios"]}
    return scenarios_dict


@pytest.fixture(scope="session")
def end_scenarios():
    """
    Load late game scenarios from hanabi_end/scenarios.json.

    Returns:
        Dict[str, dict]: Mapping of scenario_name to scenario dict
    """
    scenarios_path = Path(__file__).parent.parent / "hanabi_end" / "scenarios.json"

    with open(scenarios_path, 'r') as f:
        data = json.load(f)

    scenarios_dict = {s["scenario_name"]: s for s in data["scenarios"]}
    return scenarios_dict


@pytest.fixture(scope="session")
def turn_28_bug_scenario():
    """
    Load critical Turn 28 bug scenario.

    Returns:
        dict: Turn 28 bug scenario with detailed ground truth
    """
    bug_path = Path(__file__).parent.parent / "hanabi_end" / "bug_scenarios" / "turn_28_white_stack.json"

    with open(bug_path, 'r') as f:
        scenario = json.load(f)

    return scenario


# ============================================================================
# ABLATION STUDY FIXTURES
# ============================================================================

@pytest.fixture(scope="session")
def metrics_collector():
    """
    Shared metrics collector for all ablation tests.

    This collector accumulates results across all test runs and is used
    to generate the final comparison report.

    Returns:
        MetricsCollector: Global metrics collector instance
    """
    return _metrics_collector


@pytest.fixture
def agent_decision():
    """
    Agent decision function for ablation tests.

    Toggle between MOCK mode (for infrastructure testing) and REAL mode
    (for actual research results) using the USE_REAL_AGENT flag below.

    Returns:
        Callable: Function that takes (scenario, nl_belief_context) and returns (action, reasoning)
    """
    # ======================================================================
    # TOGGLE THIS FLAG TO SWITCH BETWEEN MOCK AND REAL AGENT
    # ======================================================================
    USE_REAL_AGENT = True  # Using real Gemini API agent
    # ======================================================================

    if USE_REAL_AGENT:
        # REAL AGENT MODE - Use actual LLM agent for research
        from tests.shared.agent_adapter import create_agent_decision_function

        # Create real agent decision function
        # Uses NL belief injection (fixed in agent_adapter.py)
        decide_fn = create_agent_decision_function(agent_type="probabilistic")
        return decide_fn

    else:
        # MOCK MODE - Return ground truth actions for infrastructure testing
        def _decide(scenario, nl_belief_context):
            """
            Mock decision function.

            Args:
                scenario: Game scenario dict
                nl_belief_context: Natural language belief description string

            Returns:
                tuple: (action_dict, reasoning_text)
            """
            action = scenario["ground_truth_best_action"].copy()

            # Mock reasoning that demonstrates good reasoning elements
            reasoning = f"""**PART 1: SITUATION ANALYSIS**
The fireworks stacks are at: Red={scenario['fireworks']['R']}, Yellow={scenario['fireworks']['Y']},
Green={scenario['fireworks']['G']}, White={scenario['fireworks']['W']}, Blue={scenario['fireworks']['B']}.

Life tokens: {scenario['life_tokens']}, Info tokens: {scenario.get('info_tokens', 8)},
Deck size: {scenario.get('deck_size', 50)}

**PART 2: IMPOSSIBILITY CONSTRAINTS**
Based on the stack heights, certain cards are impossible to be in my hand:
{scenario.get('impossibility_statements', 'No specific impossibility constraints listed.')}

**PART 3: DECISION**
Based on the game state analysis, I will {action['type'].lower()}.

Reasoning: {action.get('reasoning', 'Ground truth action for this scenario')}
"""

            return action, reasoning

        return _decide


# ============================================================================
# LEGACY FIXTURES (kept for backward compatibility)
# ============================================================================

@pytest.fixture
def turn_28_game_state():
    """
    Exact game state from Turn 28 bug.

    Returns:
        dict: Game state with fireworks, tokens, deck info
    """
    return {
        "fireworks": {
            "R": 3,
            "Y": 2,
            "G": 0,
            "W": 3,  # White stack at 3 - this is the critical state
            "B": 1
        },
        "life_tokens": 3,
        "info_tokens": 7,
        "deck_size": 15,
        "description": "Bug scenario - White-1/2/3 impossible (stack at 3)"
    }


@pytest.fixture
def turn_28_expected_beliefs():
    """
    Expected beliefs after Turn 28 card draw.

    All cards already on stacks MUST have P=0%.

    Returns:
        dict: Lists of impossible and possible (color, rank) tuples
    """
    return {
        "impossible_cards": [
            ("red", 1), ("red", 2), ("red", 3),
            ("yellow", 1), ("yellow", 2),
            ("white", 1), ("white", 2), ("white", 3),  # Bug test - these are impossible
            ("blue", 1)
        ],
        "possible_cards": [
            ("red", 4), ("red", 5),
            ("yellow", 3), ("yellow", 4), ("yellow", 5),
            ("green", 1), ("green", 2), ("green", 3), ("green", 4), ("green", 5),
            ("white", 4), ("white", 5),
            ("blue", 2), ("blue", 3), ("blue", 4), ("blue", 5)
        ]
    }


# ============================================================================
# PYTEST HOOKS
# ============================================================================

def pytest_sessionfinish(session, exitstatus):
    """
    Pytest hook called after all tests complete.

    Automatically generates the ablation study comparison report and exports
    raw data to JSON.

    Args:
        session: pytest session object
        exitstatus: Exit status code
    """
    # Only generate report if tests were run (not --collect-only)
    if session.config.option.collectonly:
        return

    # Only generate if we have results
    if not _metrics_collector.results:
        print("\n" + "=" * 70)
        print("No ablation study results collected.")
        print("Run tests with: pytest tests/hanabi_early/ tests/hanabi_mid/ tests/hanabi_end/")
        print("=" * 70 + "\n")
        return

    print("\n" + "=" * 70)
    print("GENERATING NATURAL LANGUAGE BELIEF ABLATION STUDY REPORT...")
    print("=" * 70)

    # Generate markdown report
    try:
        report = _metrics_collector.generate_stage_comparison_report()

        # Save report
        report_dir = Path(__file__).parent.parent
        report_path = report_dir / "nl_belief_ablation_results.md"

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"✅ Markdown report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️  Error generating markdown report: {e}")
        import traceback
        traceback.print_exc()

    # Export raw data to JSON
    try:
        json_dir = report_dir / "testing_results"
        json_dir.mkdir(parents=True, exist_ok=True)
        json_path = json_dir / "nl_belief_ablation_results.json"

        _metrics_collector.export_to_json(str(json_path))

        print(f"✅ Raw data exported to: {json_path}")
    except Exception as e:
        print(f"⚠️  Error exporting JSON data: {e}")
        import traceback
        traceback.print_exc()

    # Print summary statistics
    try:
        summary = _metrics_collector.generate_summary_stats()

        print("\n" + "-" * 70)
        print("SUMMARY STATISTICS:")
        print("-" * 70)
        print(f"Total tests run: {summary['total_tests_run']}")
        print(f"Scenarios tested: {summary['unique_scenarios']}")
        print(f"Game stages tested: {', '.join(summary['unique_stages'])}")

        print("\nStage Rankings (by safety):")
        for rank, stage in enumerate(summary['stage_rankings_by_safety'], 1):
            perf = summary['stage_performance'][stage]
            safe_pct = perf['safety_breakdown']['safe_percentage']
            print(f"  {rank}. {stage}: {safe_pct:.1f}% safe actions")

        # Print Turn 28 bug result if tested
        turn_28_results = [r for r in _metrics_collector.results if r.scenario_name == "turn_28_white_stack_bug"]
        if turn_28_results:
            result = turn_28_results[0]
            bug_status = "✅ PASSED" if result.action_safety != "FATAL" else "❌ FAILED"
            print(f"\nTurn 28 Bug Test: {bug_status}")
            print(f"  Action: {result.action_type} target={result.action_target}")
            print(f"  Safety: {result.action_safety}")
            if result.judge_score is not None:
                print(f"  Judge Score: {result.judge_score:.2f}/1.0")

    except Exception as e:
        print(f"⚠️  Error generating summary: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"Ablation study complete!")
    try:
        print(f"View full report: cat {report_path}")
    except:
        pass
    print("=" * 70 + "\n")
