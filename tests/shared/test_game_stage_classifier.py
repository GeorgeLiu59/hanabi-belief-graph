"""Unit tests for hybrid game stage classifier.

Tests all edge cases enumerated in GAME_STAGE_DEFINITION.md:
- Case 1: Early turn, exceptional progress
- Case 2: Mid-game resources, critical pressure
- Case 3: Late turn, minimal progress
- Case 4: Boundary threshold exactly met
- Case 5: Turn 28 bug scenario

Also tests:
- Boundary conditions (t=8/9, c=5/6, etc.)
- Priority ordering (late > mid > early)
- Basic classification correctness
"""

import pytest
from typing import Dict, Any

from .game_stage_classifier import classify_game_stage, get_classification_metadata


class TestBasicClassification:
    """Test straightforward classification cases."""

    def test_early_game_turn_1(self):
        """Turn 1 - clean early game state."""
        scenario = {
            'turn': 1,
            'fireworks': {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0},
            'life_tokens': 3,
            'info_tokens': 8
        }
        assert classify_game_stage(scenario) == 'early_game'

    def test_mid_game_typical(self):
        """Turn 12 - typical mid game state."""
        scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},
            'life_tokens': 2,
            'info_tokens': 4
        }
        assert classify_game_stage(scenario) == 'mid_game'

    def test_late_game_turn_28(self):
        """Turn 28 - late game by turn count."""
        scenario = {
            'turn': 28,
            'fireworks': {'R': 3, 'Y': 2, 'G': 0, 'W': 3, 'B': 1},
            'life_tokens': 3,
            'info_tokens': 7
        }
        assert classify_game_stage(scenario) == 'late_game'


class TestEdgeCases:
    """Test edge cases from formal specification."""

    def test_case_1_early_turn_exceptional_progress(self):
        """
        Edge Case 1: Early turn, high progress.

        State: t=5, c=10, L=3, I=8, D=45
        Expected: early_game (by elimination)
        Rationale: Low turn despite high progress, still low pressure
        """
        scenario = {
            'turn': 5,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 2, 'B': 2},  # c=10
            'life_tokens': 3,
            'info_tokens': 8,
            'deck_size': 45
        }
        assert classify_game_stage(scenario) == 'early_game'

    def test_case_2_mid_resources_critical_pressure(self):
        """
        Edge Case 2: Mid-game resources, critical pressure.

        State: t=12, c=8, L=1, I=5, D=35
        Expected: late_game (pressure override)
        Rationale: L=1 triggers late despite mid temporal/progress
        """
        scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'life_tokens': 1,  # Critical trigger
            'info_tokens': 5,
            'deck_size': 35
        }
        assert classify_game_stage(scenario) == 'late_game'

    def test_case_3_late_turn_minimal_progress(self):
        """
        Edge Case 3: Late turn, minimal progress.

        State: t=25, c=3, L=3, I=7, D=40
        Expected: late_game (temporal trigger)
        Rationale: Turn 25 indicates prolonged play
        """
        scenario = {
            'turn': 25,
            'fireworks': {'R': 1, 'Y': 1, 'G': 1, 'W': 0, 'B': 0},  # c=3
            'life_tokens': 3,
            'info_tokens': 7,
            'deck_size': 40
        }
        assert classify_game_stage(scenario) == 'late_game'

    def test_case_4_boundary_threshold_exact(self):
        """
        Edge Case 4: Boundary threshold exactly met.

        State: t=9, c=6, L=2, I=2, D=30
        Expected: mid_game (all conditions satisfied)
        Note: Thresholds are inclusive (≥, ≤)
        """
        scenario = {
            'turn': 9,  # Exactly at mid lower bound
            'fireworks': {'R': 2, 'Y': 1, 'G': 1, 'W': 1, 'B': 1},  # c=6 exactly
            'life_tokens': 2,  # Exactly at mid lower bound
            'info_tokens': 2,  # Exactly at mid lower bound
            'deck_size': 30
        }
        assert classify_game_stage(scenario) == 'mid_game'

    def test_case_5_turn_28_bug_scenario(self):
        """
        Edge Case 5: Turn 28 bug scenario.

        State: t=28, c=9, L=3, I=7, D=40, β=true
        Expected: late_game (temporal + bug flag)
        Special: Critical research test case
        """
        scenario = {
            'turn': 28,
            'fireworks': {'R': 3, 'Y': 2, 'G': 0, 'W': 3, 'B': 1},  # c=9
            'life_tokens': 3,
            'info_tokens': 7,
            'deck_size': 40,
            'is_bug_scenario': True  # Explicit flag
        }
        assert classify_game_stage(scenario) == 'late_game'


class TestBoundaryConditions:
    """Test threshold boundaries for all dimensions."""

    def test_turn_boundary_8_vs_9(self):
        """Turn 8 (early) vs Turn 9 (mid)."""
        base_scenario = {
            'fireworks': {'R': 2, 'Y': 1, 'G': 1, 'W': 1, 'B': 1},  # c=6
            'life_tokens': 2,
            'info_tokens': 2,
            'deck_size': 35
        }

        # Turn 8: not in mid range (9-20), not in late (≥20)
        scenario_8 = {**base_scenario, 'turn': 8}
        assert classify_game_stage(scenario_8) == 'early_game'

        # Turn 9: exactly at mid lower bound
        scenario_9 = {**base_scenario, 'turn': 9}
        assert classify_game_stage(scenario_9) == 'mid_game'

    def test_turn_boundary_19_vs_20(self):
        """Turn 19 (could be mid) vs Turn 20 (late)."""
        base_scenario = {
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'life_tokens': 2,
            'info_tokens': 3,
            'deck_size': 35
        }

        # Turn 19: in mid range, satisfies all mid conditions
        scenario_19 = {**base_scenario, 'turn': 19}
        assert classify_game_stage(scenario_19) == 'mid_game'

        # Turn 20: exactly at late temporal trigger
        scenario_20 = {**base_scenario, 'turn': 20}
        assert classify_game_stage(scenario_20) == 'late_game'

    def test_cards_played_boundary_14_vs_15(self):
        """Cards played 14 (could be mid) vs 15 (late)."""
        base_scenario = {
            'turn': 15,
            'life_tokens': 2,
            'info_tokens': 3,
            'deck_size': 30
        }

        # c=14: in mid range (6-15)
        scenario_14 = {
            **base_scenario,
            'fireworks': {'R': 3, 'Y': 3, 'G': 3, 'W': 3, 'B': 2}  # c=14
        }
        assert classify_game_stage(scenario_14) == 'mid_game'

        # c=15: exactly at late progress trigger
        scenario_15 = {
            **base_scenario,
            'fireworks': {'R': 3, 'Y': 3, 'G': 3, 'W': 3, 'B': 3}  # c=15
        }
        assert classify_game_stage(scenario_15) == 'late_game'

    def test_life_tokens_boundary_1_vs_2(self):
        """Life tokens 1 (late) vs 2 (could be mid)."""
        base_scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'info_tokens': 4,
            'deck_size': 35
        }

        # L=1: critical pressure, triggers late
        scenario_1 = {**base_scenario, 'life_tokens': 1}
        assert classify_game_stage(scenario_1) == 'late_game'

        # L=2: satisfies mid condition
        scenario_2 = {**base_scenario, 'life_tokens': 2}
        assert classify_game_stage(scenario_2) == 'mid_game'

    def test_info_tokens_boundary_1_vs_2(self):
        """Info tokens 1 (late) vs 2 (could be mid)."""
        base_scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'life_tokens': 2,
            'deck_size': 35
        }

        # I=1: critical resource scarcity, triggers late
        scenario_1 = {**base_scenario, 'info_tokens': 1}
        assert classify_game_stage(scenario_1) == 'late_game'

        # I=2: satisfies mid condition
        scenario_2 = {**base_scenario, 'info_tokens': 2}
        assert classify_game_stage(scenario_2) == 'mid_game'

    def test_deck_size_boundary_10_vs_11(self):
        """Deck size 10 (late) vs 11 (not late trigger)."""
        base_scenario = {
            'turn': 15,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'life_tokens': 2,
            'info_tokens': 3
        }

        # D=10: exactly at late deck trigger
        scenario_10 = {**base_scenario, 'deck_size': 10}
        assert classify_game_stage(scenario_10) == 'late_game'

        # D=11: not a late trigger, satisfies mid
        scenario_11 = {**base_scenario, 'deck_size': 11}
        assert classify_game_stage(scenario_11) == 'mid_game'


class TestPriorityLogic:
    """Test that priority ordering works correctly (late > mid > early)."""

    def test_late_overrides_mid(self):
        """State satisfies mid conditions but has late trigger."""
        scenario = {
            'turn': 15,  # In mid range (9-20)
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8 in mid range
            'life_tokens': 1,  # Late trigger!
            'info_tokens': 4,  # Satisfies mid (≥2)
            'deck_size': 35
        }
        # Despite satisfying mid temporal/progress, L=1 forces late
        assert classify_game_stage(scenario) == 'late_game'

    def test_multiple_late_triggers(self):
        """Multiple late triggers (should still just return late_game)."""
        scenario = {
            'turn': 25,  # Late trigger (≥20)
            'fireworks': {'R': 5, 'Y': 4, 'G': 3, 'W': 3, 'B': 2},  # c=17, late trigger (≥15)
            'life_tokens': 1,  # Late trigger
            'info_tokens': 0,  # Late trigger (≤1)
            'deck_size': 5  # Late trigger (≤10)
        }
        assert classify_game_stage(scenario) == 'late_game'

    def test_mid_requires_all_conditions(self):
        """Mid game fails if ANY condition not met."""
        base_mid = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},  # c=8
            'life_tokens': 2,
            'info_tokens': 4
        }

        # All conditions met → mid
        assert classify_game_stage(base_mid) == 'mid_game'

        # Fail turn condition (too early)
        fail_turn = {**base_mid, 'turn': 8}
        assert classify_game_stage(fail_turn) == 'early_game'

        # Fail progress condition (too few cards)
        fail_progress = {
            **base_mid,
            'fireworks': {'R': 1, 'Y': 1, 'G': 1, 'W': 0, 'B': 0}  # c=3
        }
        assert classify_game_stage(fail_progress) == 'early_game'


class TestMetadataFunction:
    """Test the get_classification_metadata helper."""

    def test_metadata_structure(self):
        """Verify metadata includes all expected fields."""
        scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},
            'life_tokens': 2,
            'info_tokens': 4
        }

        metadata = get_classification_metadata(scenario)

        # Check structure
        assert 'stage' in metadata
        assert 'state_variables' in metadata
        assert 'triggers' in metadata
        assert 'interpretation' in metadata

        # Check state variables
        assert metadata['state_variables']['t'] == 12
        assert metadata['state_variables']['c'] == 8
        assert metadata['state_variables']['L'] == 2
        assert metadata['state_variables']['I'] == 4

    def test_metadata_late_triggers(self):
        """Verify late triggers are correctly identified."""
        scenario = {
            'turn': 25,  # Late trigger
            'fireworks': {'R': 1, 'Y': 1, 'G': 1, 'W': 0, 'B': 0},
            'life_tokens': 1,  # Late trigger
            'info_tokens': 5
        }

        metadata = get_classification_metadata(scenario)

        assert metadata['stage'] == 'late_game'
        assert len(metadata['triggers']['late_triggers']) == 2
        assert any('turn' in t for t in metadata['triggers']['late_triggers'])
        assert any('life' in t for t in metadata['triggers']['late_triggers'])


class TestFieldNameCompatibility:
    """Test handling of different field name conventions."""

    def test_info_tokens_vs_information_tokens(self):
        """Handle both 'info_tokens' and 'information_tokens'."""
        base_scenario = {
            'turn': 12,
            'fireworks': {'R': 2, 'Y': 2, 'G': 2, 'W': 1, 'B': 1},
            'life_tokens': 2
        }

        # Using 'info_tokens'
        scenario_1 = {**base_scenario, 'info_tokens': 4}
        assert classify_game_stage(scenario_1) == 'mid_game'

        # Using 'information_tokens' (alternative naming)
        scenario_2 = {**base_scenario, 'information_tokens': 4}
        assert classify_game_stage(scenario_2) == 'mid_game'

        # Both present (info_tokens takes precedence)
        scenario_3 = {**base_scenario, 'info_tokens': 4, 'information_tokens': 1}
        assert classify_game_stage(scenario_3) == 'mid_game'  # Uses 4, not 1


if __name__ == '__main__':
    # Allow running tests directly
    pytest.main([__file__, '-v'])
