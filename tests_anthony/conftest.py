"""
pytest fixtures for Hanabi belief graph tests.

Provides reusable test fixtures for:
- Game environments
- Agent instances
- Turn 28 bug scenario
- Multi-turn progression scenarios
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


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
