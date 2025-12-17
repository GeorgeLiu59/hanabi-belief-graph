"""Hybrid game stage classification for Hanabi ablation study.

Implements the formal specification from GAME_STAGE_DEFINITION.md.

Classification function:
    γ: S → {early_game, mid_game, late_game}

Where game state S = (t, c, L, I, D, β):
    t = turn number
    c = cards played (sum of fireworks)
    L = life tokens
    I = info tokens
    D = deck size
    β = bug scenario flag

See GAME_STAGE_DEFINITION.md for complete formal specification and rationale.
"""

from typing import Dict, Any, Literal

# Type alias for game stages
GameStage = Literal["early_game", "mid_game", "late_game"]


def classify_game_stage(scenario: Dict[str, Any]) -> GameStage:
    """
    Classify Hanabi game state into strategic phase.

    Uses hybrid approach combining temporal, progress, and pressure dimensions.
    Priority logic: late > mid > early (any pressure trigger → late game).

    Args:
        scenario: Game state dictionary containing:
            - turn: int (turn number, ≥ 1)
            - fireworks: Dict[str, int] (stack heights for R/Y/G/W/B)
            - life_tokens: int (0-3)
            - info_tokens: int (0-8, often named 'information_tokens' in some schemas)
            - deck_size: int (0-50, optional, default=50)
            - is_bug_scenario: bool (optional, default=False)

    Returns:
        GameStage: One of "early_game", "mid_game", "late_game"

    Raises:
        KeyError: If required fields missing from scenario
        AssertionError: If field values out of valid range (in debug mode)

    Examples:
        >>> classify_game_stage({
        ...     'turn': 1,
        ...     'fireworks': {'R':0, 'Y':0, 'G':0, 'W':0, 'B':0},
        ...     'life_tokens': 3,
        ...     'info_tokens': 8
        ... })
        'early_game'

        >>> classify_game_stage({
        ...     'turn': 12,
        ...     'fireworks': {'R':2, 'Y':2, 'G':2, 'W':1, 'B':1},
        ...     'life_tokens': 2,
        ...     'info_tokens': 4
        ... })
        'mid_game'

        >>> classify_game_stage({
        ...     'turn': 28,
        ...     'fireworks': {'R':3, 'Y':2, 'G':0, 'W':3, 'B':1},
        ...     'life_tokens': 3,
        ...     'info_tokens': 7,
        ...     'is_bug_scenario': True
        ... })
        'late_game'

    Formal Specification:
        Late Game (Priority 1 - any trigger):
            γ(s) = late ⟺ t ≥ 20 ∨ c ≥ 15 ∨ L = 1 ∨ I ≤ 1 ∨ D ≤ 10 ∨ β = true

        Mid Game (Priority 2 - all conditions):
            γ(s) = mid ⟺ (9 ≤ t ≤ 20) ∧ (6 ≤ c ≤ 15) ∧ (L ≥ 2) ∧ (I ≥ 2) ∧ ¬late(s)

        Early Game (Default):
            γ(s) = early ⟺ ¬late(s) ∧ ¬mid(s)
    """
    # Extract state variables
    t = scenario.get('turn', 1)
    fireworks = scenario['fireworks']
    c = sum(fireworks.values())  # Total cards played
    L = scenario['life_tokens']

    # Handle both 'info_tokens' and 'information_tokens' field names
    I = scenario.get('info_tokens', scenario.get('information_tokens', 8))

    D = scenario.get('deck_size', 50)
    β = scenario.get('is_bug_scenario', False)

    # Validation (assertions for development/testing)
    assert t >= 1, f"Invalid turn number: {t} (must be ≥ 1)"
    assert 0 <= c <= 25, f"Invalid cards played: {c} (must be 0-25)"
    assert L in {0, 1, 2, 3}, f"Invalid life tokens: {L} (must be 0, 1, 2, or 3)"
    assert 0 <= I <= 8, f"Invalid info tokens: {I} (must be 0-8)"
    assert 0 <= D <= 50, f"Invalid deck size: {D} (must be 0-50)"

    # LATE GAME - Priority 1 (any trigger → late)
    # High-pressure situations override everything else
    if (t >= 20 or c >= 15 or L == 1 or I <= 1 or D <= 10 or β):
        return 'late_game'

    # MID GAME - Priority 2 (all conditions must hold)
    # Stable building phase with moderate progress
    if (9 <= t <= 20 and 6 <= c <= 15 and L >= 2 and I >= 2):
        return 'mid_game'

    # EARLY GAME - Default
    # Low-pressure exploration phase
    return 'early_game'


def get_classification_metadata(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get detailed classification metadata for debugging/analysis.

    Args:
        scenario: Game state dictionary (same as classify_game_stage)

    Returns:
        Dictionary containing:
        {
            'stage': str (early_game / mid_game / late_game),
            'state_variables': {
                't': int,
                'c': int,
                'L': int,
                'I': int,
                'D': int,
                'β': bool
            },
            'triggers': {
                'late_triggers': List[str],  # Which late conditions triggered
                'mid_conditions': Dict[str, bool],  # Mid game condition checks
            },
            'interpretation': str  # Human-readable explanation
        }
    """
    # Extract state variables
    t = scenario.get('turn', 1)
    fireworks = scenario['fireworks']
    c = sum(fireworks.values())
    L = scenario['life_tokens']
    I = scenario.get('info_tokens', scenario.get('information_tokens', 8))
    D = scenario.get('deck_size', 50)
    β = scenario.get('is_bug_scenario', False)

    # Check late game triggers
    late_triggers = []
    if t >= 20:
        late_triggers.append(f"turn ≥ 20 (t={t})")
    if c >= 15:
        late_triggers.append(f"cards_played ≥ 15 (c={c})")
    if L == 1:
        late_triggers.append(f"life_tokens = 1 (L={L})")
    if I <= 1:
        late_triggers.append(f"info_tokens ≤ 1 (I={I})")
    if D <= 10:
        late_triggers.append(f"deck_size ≤ 10 (D={D})")
    if β:
        late_triggers.append("bug_scenario = true")

    # Check mid game conditions
    mid_conditions = {
        '9 ≤ turn ≤ 20': 9 <= t <= 20,
        '6 ≤ cards_played ≤ 15': 6 <= c <= 15,
        'life_tokens ≥ 2': L >= 2,
        'info_tokens ≥ 2': I >= 2
    }

    # Classify
    stage = classify_game_stage(scenario)

    # Generate interpretation
    if stage == 'late_game':
        interpretation = f"Late game (pressure): {', '.join(late_triggers)}"
    elif stage == 'mid_game':
        interpretation = f"Mid game (building phase): all conditions satisfied"
    else:  # early_game
        if late_triggers:
            interpretation = f"Early game (would be late but: {late_triggers[0]})"
        else:
            failed_mid = [k for k, v in mid_conditions.items() if not v]
            if failed_mid:
                interpretation = f"Early game (mid conditions failed: {', '.join(failed_mid)})"
            else:
                interpretation = "Early game (default - low pressure)"

    return {
        'stage': stage,
        'state_variables': {
            't': t,
            'c': c,
            'L': L,
            'I': I,
            'D': D,
            'β': β
        },
        'triggers': {
            'late_triggers': late_triggers,
            'mid_conditions': mid_conditions
        },
        'interpretation': interpretation
    }
