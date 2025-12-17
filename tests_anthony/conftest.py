"""
pytest fixtures for Hanabi belief graph tests.

Provides reusable test fixtures for:
- Game environments
- Agent instances
- Turn 28 bug scenario
- Multi-turn progression scenarios
- Ablation study infrastructure (metrics collection, session-level reporting)
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import ablation study components
from .metrics import MetricsCollector

# Global metrics collector shared across all tests
_metrics_collector = MetricsCollector()


@pytest.fixture
def hanabi_env():
    """
    Create fresh 2-player Hanabi-Full-CardKnowledge environment.

    Returns:
        rl_env.Environment: Fresh Hanabi environment
    """
    pytest.skip("Not implemented - requires hanabi_learning_environment import")


@pytest.fixture
def probabilistic_agent():
    """
    Create probabilistic belief graph agent for testing.

    Returns:
        BeliefGraphProbabilisticAgent: Fresh agent instance
    """
    pytest.skip("Not implemented - requires agent import")


@pytest.fixture
def turn_28_game_state():
    """
    Exact game state from Turn 28 bug.

    Source: Game logs lines 20590-20764
    Bug: Agent played White-1 when White stack at 3

    Returns:
        dict: Game state with fireworks, tokens, deck info
    """
    return {
        "fireworks": {
            "R": 3,  # Red stack: 1, 2, 3 played
            "Y": 2,  # Yellow stack: 1, 2 played
            "G": 0,  # Green stack: nothing played
            "W": 3,  # White stack: 1, 2, 3 played
            "B": 1   # Blue stack: 1 played
        },
        "life_tokens": 3,
        "info_tokens": 7,
        "deck_size": 40,
        "discard_pile": [],  # TODO: Extract from logs if needed
        "description": "Bug scenario - White-1 played when White stack at 3"
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
            # Red stack at 3 → R1, R2, R3 impossible
            ("red", 1), ("red", 2), ("red", 3),
            # Yellow stack at 2 → Y1, Y2 impossible
            ("yellow", 1), ("yellow", 2),
            # Green stack at 0 → nothing impossible yet
            # White stack at 3 → W1, W2, W3 impossible
            ("white", 1), ("white", 2), ("white", 3),
            # Blue stack at 1 → B1 impossible
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


@pytest.fixture
def turn_progression_scenarios():
    """
    Game states at different turns for parametrized testing.

    Tests belief evolution as game progresses from turn 1 to turn 28.

    Returns:
        list[dict]: List of game state scenarios
    """
    return [
        {
            "turn": 1,
            "fireworks": {"R": 0, "Y": 0, "G": 0, "W": 0, "B": 0},
            "impossible_cards": [],
            "description": "Game start - all cards possible"
        },
        {
            "turn": 5,
            "fireworks": {"R": 1, "Y": 0, "G": 0, "W": 0, "B": 0},
            "impossible_cards": [("red", 1)],
            "description": "Early game - first card played"
        },
        {
            "turn": 10,
            "fireworks": {"R": 2, "Y": 1, "G": 0, "W": 1, "B": 0},
            "impossible_cards": [
                ("red", 1), ("red", 2),
                ("yellow", 1),
                ("white", 1)
            ],
            "description": "Mid-early - multiple stacks building"
        },
        {
            "turn": 15,
            "fireworks": {"R": 3, "Y": 2, "G": 1, "W": 2, "B": 1},
            "impossible_cards": [
                ("red", 1), ("red", 2), ("red", 3),
                ("yellow", 1), ("yellow", 2),
                ("green", 1),
                ("white", 1), ("white", 2),
                ("blue", 1)
            ],
            "description": "Mid-game - significant progress"
        },
        {
            "turn": 28,
            "fireworks": {"R": 3, "Y": 2, "G": 0, "W": 3, "B": 1},
            "impossible_cards": [
                ("red", 1), ("red", 2), ("red", 3),
                ("yellow", 1), ("yellow", 2),
                ("white", 1), ("white", 2), ("white", 3),
                ("blue", 1)
            ],
            "description": "BUG SCENARIO - White stack at 3"
        }
    ]


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
def mock_agent_decision():
    """
    Agent decision function for ablation tests.

    Toggle between MOCK mode (for infrastructure testing) and REAL mode
    (for actual research results) using the USE_REAL_AGENT flag below.

    Returns:
        Callable: Function that takes (scenario, context) and returns (action, reasoning)
    """
    # ======================================================================
    # TOGGLE THIS FLAG TO SWITCH BETWEEN MOCK AND REAL AGENT
    # ======================================================================
    USE_REAL_AGENT = True  # Set to True to use real LLM agent
    # ======================================================================

    if USE_REAL_AGENT:
        # REAL AGENT MODE - Use actual LLM agent for research
        from .agent_adapter import create_agent_decision_function

        # Create real agent decision function
        # TODO: Fix agent_adapter to properly extract reasoning
        decide_fn = create_agent_decision_function(agent_type="probabilistic")
        return decide_fn

    else:
        # MOCK MODE - Return ground truth actions for infrastructure testing
        def _decide(scenario, context):
            """
            Mock decision function.

            Args:
                scenario: Game scenario dict
                context: Context string (formatted according to context type)

            Returns:
                tuple: (action_dict, reasoning_text)
            """
            action = scenario["ground_truth_best_action"].copy()

            reasoning = f"""REASONING:

**PART 1: SITUATION ANALYSIS**
The fireworks stacks are at: Red={scenario['fireworks']['R']}, Yellow={scenario['fireworks']['Y']},
Green={scenario['fireworks']['G']}, White={scenario['fireworks']['W']}, Blue={scenario['fireworks']['B']}.

**PART 2: DECISION**
Based on the game state, I will {action['type'].lower()}.

Reasoning: {action.get('reasoning', 'Ground truth action')}
"""

            return action, reasoning

        return _decide


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
        print("Run ablation tests with: pytest tests_anthony/test_context_ablation.py")
        print("=" * 70 + "\n")
        return

    print("\n" + "=" * 70)
    print("GENERATING ABLATION STUDY REPORT...")
    print("=" * 70)

    # Generate markdown report
    try:
        report = _metrics_collector.generate_comparison_report()

        # Save report
        report_dir = Path(__file__).parent
        report_path = report_dir / "ablation_results.md"

        with open(report_path, 'w') as f:
            f.write(report)

        print(f"✅ Markdown report saved to: {report_path}")
    except Exception as e:
        print(f"⚠️  Error generating markdown report: {e}")

    # Export raw data to JSON
    try:
        json_path = Path(__file__).parent / "ablation_results.json"
        _metrics_collector.export_to_json(str(json_path))

        print(f"✅ Raw data exported to: {json_path}")
    except Exception as e:
        print(f"⚠️  Error exporting JSON data: {e}")

    # Print summary statistics
    try:
        summary = _metrics_collector.generate_summary_stats()

        print("\n" + "-" * 70)
        print("SUMMARY STATISTICS:")
        print("-" * 70)
        print(f"Total tests run: {summary['total_tests_run']}")
        print(f"Scenarios tested: {summary['unique_scenarios']}")
        print(f"Context types compared: {summary['unique_contexts']}")

        print("\nContext Rankings (by safety):")
        for rank, context in enumerate(summary['context_rankings_by_safety'], 1):
            perf = summary['context_performance'][context]
            safe_pct = perf['safety_breakdown']['safe_percentage']
            print(f"  {rank}. {context}: {safe_pct:.1f}% safe")

    except Exception as e:
        print(f"⚠️  Error generating summary: {e}")

    print("\n" + "=" * 70)
    print(f"Ablation study complete!")
    print(f"View full report: cat {report_path}")
    print("=" * 70 + "\n")
