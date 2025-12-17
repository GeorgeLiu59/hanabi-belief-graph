# Hybrid Game Stage Classification: Formal Specification

## 1. Formal Problem Statement

**Purpose**: Provide a deterministic, reproducible classification system for categorizing Hanabi game states into strategically meaningful phases.

**Research Motivation**: Decision-making requirements fundamentally differ across game phases. Isolating these phases enables controlled experiments on how information density and pressure affect LLM reasoning quality.

## 2. Mathematical Definition

### 2.1 State Space

Game state `s ∈ S` is defined by the tuple:

```
s = (t, c, L, I, D, β)
```

Where:
- `t` ∈ [1, ∞): turn number (temporal dimension)
- `c` ∈ [0, 25]: cards successfully played to fireworks (progress dimension)
- `L` ∈ {0, 1, 2, 3}: life tokens remaining (pressure dimension)
- `I` ∈ [0, 8]: information tokens remaining (pressure dimension)
- `D` ∈ [0, 50]: cards remaining in deck (temporal/pressure dimension)
- `β` ∈ {true, false}: bug scenario flag (experimental control)

### 2.2 Classification Function

```
γ: S → {early_game, mid_game, late_game}
```

**Priority-based evaluation** (checked in order):

1. **Late Game** (Priority 1):
   ```
   γ(s) = late_game ⟺ t ≥ 20 ∨ c ≥ 15 ∨ L = 1 ∨ I ≤ 1 ∨ D ≤ 10 ∨ β = true
   ```

2. **Mid Game** (Priority 2):
   ```
   γ(s) = mid_game ⟺ (9 ≤ t ≤ 20) ∧ (6 ≤ c ≤ 15) ∧ (L ≥ 2) ∧ (I ≥ 2) ∧ ¬late(s)
   ```

3. **Early Game** (Default):
   ```
   γ(s) = early_game ⟺ ¬late(s) ∧ ¬mid(s)
   ```

## 3. Design Rationale

### 3.1 Hybrid Approach Justification

**Single-dimension approaches fail to capture strategic reality:**

| Approach | Strength | Critical Failure Case |
|----------|----------|----------------------|
| Turn-based only | Simple, deterministic | Turn 5 with 1 life ≠ Turn 5 with 3 lives |
| Progress-based only | Reflects advancement | Score 10 with 0 info ≠ Score 10 with 8 info |
| Pressure-based only | Identifies risk | Ambiguous thresholds, subjective |

**Hybrid solution**: Combines temporal, progress, AND pressure dimensions with priority logic.

### 3.2 Dimensional Analysis

**Temporal Dimension** (`t`):
- **Captures**: Game progression independent of performance quality
- **Justification**: Turn count correlates with information revelation (cards seen/discarded)
- **Threshold rationale**: Turn 20 ≈ 40% through typical 50-card deck

**Progress Dimension** (`c`):
- **Captures**: Actual advancement toward win condition (score 25)
- **Justification**: Higher score → more constraints on remaining playable cards
- **Threshold rationale**: 15 cards ≈ 60% game completion, critical card tracking essential

**Pressure Dimension** (`L`, `I`, `D`):
- **Captures**: Resource scarcity forcing strategic pivots
- **Justification**: Single pressure trigger changes decision calculus fundamentally
- **Threshold rationale**:
  - `L = 1`: No margin for error (one mistake ends game)
  - `I ≤ 1`: Cannot give meaningful hints (coordination breaks down)
  - `D ≤ 10`: Endgame countdown active (limited draws remaining)

### 3.3 Priority Logic Justification

**Why Late > Mid > Early?**

Decision tree ensures **conservative classification** under uncertainty:
- **Late game override**: ANY critical condition → late (pressure dominates)
- **Mid game conjunction**: ALL conditions must hold (stable building phase)
- **Early game default**: Low-pressure situations even if ambiguous

**Alternative rejected**: Weighted scoring (e.g., `score = 0.3t + 0.4c + 0.3pressure`)
- **Problem**: Arbitrary weights, non-interpretable, doesn't reflect strategic phase shifts

## 4. Edge Case Enumeration & Resolution

### Case 1: Early Turn, Exceptional Progress
```
State: t=5, c=10, L=3, I=8, D=45
Analysis:
  - c=10 ≥ 6 violates early_game strict condition
  - t=5 < 9 violates mid_game temporal requirement
  - No late_game triggers
Classification: early_game (by default rule)

Interpretation:
- Exceptionally efficient early play (perfect execution)
- Still low-pressure environment (full resources)
- Accept as valid edge case
```

### Case 2: Mid-Game Resources, Critical Pressure
```
State: t=12, c=8, L=1, I=5, D=35
Analysis:
  - Satisfies mid_game temporal (9 ≤ 12 ≤ 20) ✓
  - Satisfies mid_game progress (6 ≤ 8 ≤ 15) ✓
  - BUT L=1 triggers late_game (priority override)
Classification: late_game (pressure override)

Justification:
- Single life token fundamentally changes risk calculus
- Must employ conservative endgame strategy regardless of turn/progress
- Reflects strategic reality
```

### Case 3: Late Turn, Minimal Progress
```
State: t=25, c=3, L=3, I=7, D=40
Analysis:
  - t=25 ≥ 20 triggers late_game
  - Progress minimal (only 3 cards played)
  - Resources healthy (3 lives, 7 info)
Classification: late_game (temporal trigger)

Interpretation:
- Indicates prolonged suboptimal/unlucky play
- Turn 25+ means significant information revealed through discards
- Endgame pressure real despite low score
```

### Case 4: Boundary Threshold Exactly Met
```
State: t=9, c=6, L=2, I=2, D=30
Analysis:
  - ALL mid_game thresholds exactly at minimum
  - t=9, c=6, L=2, I=2 all satisfy ≥ conditions
  - No late_game triggers
Classification: mid_game (all conditions satisfied)

Note: Thresholds are inclusive (≥, ≤ operators)
```

### Case 5: Turn 28 Bug Scenario (Critical Research Test)
```
State: t=28, c=9, L=3, I=7, D=40, β=true
Analysis:
  - t=28 ≥ 20 → late_game trigger
  - β=true → late_game trigger (explicit flag)
  - Multiple triggers ensure classification
Classification: late_game (temporal + bug flag)

Special handling:
- Ground truth includes explicit impossibility statements
- LLM-as-Judge always evaluates this scenario
- Critical test for belief system correctness
```

## 5. Implementation

### 5.1 Reference Implementation

```python
from typing import Dict, Any, Literal

GameStage = Literal["early_game", "mid_game", "late_game"]

def classify_game_stage(scenario: Dict[str, Any]) -> GameStage:
    """
    Classify Hanabi game state into strategic phase.

    Args:
        scenario: Game state dictionary containing:
            - turn: int (turn number, ≥ 1)
            - fireworks: Dict[str, int] (stack heights for R/Y/G/W/B)
            - life_tokens: int (0-3)
            - info_tokens: int (0-8)
            - deck_size: int (0-50, optional, default=50)
            - is_bug_scenario: bool (optional, default=False)

    Returns:
        GameStage: One of "early_game", "mid_game", "late_game"

    Raises:
        KeyError: If required fields missing from scenario
        ValueError: If field values out of valid range

    Examples:
        >>> classify_game_stage({'turn': 1, 'fireworks': {'R':0,'Y':0,'G':0,'W':0,'B':0}, 'life_tokens': 3, 'info_tokens': 8})
        'early_game'

        >>> classify_game_stage({'turn': 12, 'fireworks': {'R':2,'Y':2,'G':2,'W':1,'B':1}, 'life_tokens': 2, 'info_tokens': 4})
        'mid_game'

        >>> classify_game_stage({'turn': 28, 'fireworks': {'R':3,'Y':2,'G':0,'W':3,'B':1}, 'life_tokens': 3, 'info_tokens': 7, 'is_bug_scenario': True})
        'late_game'
    """
    # Extract state variables
    t = scenario.get('turn', 1)
    fireworks = scenario['fireworks']
    c = sum(fireworks.values())
    L = scenario['life_tokens']
    I = scenario['info_tokens']
    D = scenario.get('deck_size', 50)
    β = scenario.get('is_bug_scenario', False)

    # Validation (optional but recommended for research rigor)
    assert 1 <= t, f"Invalid turn: {t}"
    assert 0 <= c <= 25, f"Invalid cards played: {c}"
    assert L in {0, 1, 2, 3}, f"Invalid life tokens: {L}"
    assert 0 <= I <= 8, f"Invalid info tokens: {I}"
    assert 0 <= D <= 50, f"Invalid deck size: {D}"

    # LATE GAME - Priority 1 (any trigger)
    if (t >= 20 or c >= 15 or L == 1 or I <= 1 or D <= 10 or β):
        return 'late_game'

    # MID GAME - Priority 2 (all conditions)
    if (9 <= t <= 20 and 6 <= c <= 15 and L >= 2 and I >= 2):
        return 'mid_game'

    # EARLY GAME - Default
    return 'early_game'
```

### 5.2 Invariant Checks

**Classifier must satisfy**:
1. **Totality**: Every valid state `s ∈ S` maps to exactly one stage
2. **Determinism**: Same input always produces same output
3. **Monotonicity (weak)**: Adding pressure increases likelihood of late_game classification

**Unit tests** (see `tests_anthony/shared/test_game_stage_classifier.py`):
- Verify all 5 edge cases
- Test boundary conditions (t=8/9, c=5/6, etc.)
- Validate priority ordering (late > mid > early)
- Confirm Turn 28 bug classification

## 6. Validation Strategy

### 6.1 Theoretical Validation

**Coverage analysis**:
- **Early game space**: Large (low t, low c, high L, high I)
- **Mid game space**: Moderate (bounded temporal/progress region)
- **Late game space**: Large (any pressure trigger expands space)

**Overlap**: None (priority logic ensures mutual exclusion)

### 6.2 Empirical Validation

**Method**: Apply classifier to historical game data, verify alignment with human strategic intuition

**Metrics**:
1. **Distribution**: Proportion of states in each stage (expect: early > mid > late)
2. **Stage transitions**: Verify monotonic progression (early → mid → late, no reversals)
3. **Inter-rater agreement**: Compare to human expert classifications

### 6.3 Sensitivity Analysis

**Question**: How do threshold changes affect classification?

**Test**: Vary thresholds ±2 and measure reclassification rate

| Parameter | Base | Variant | % Reclassified |
|-----------|------|---------|----------------|
| t (late) | 20 | 18 / 22 | TBD |
| c (late) | 15 | 13 / 17 | TBD |
| c (mid) | 6-15 | 5-16 / 7-14 | TBD |

**Expectation**: Classification should be robust to small threshold variations (< 10% reclassification)

## 7. Limitations & Future Work

### 7.1 Known Limitations

1. **Discrete categories**: Real gameplay may exhibit gradual transitions
   - **Mitigation**: Priority logic provides clear decision boundaries

2. **Ignores player count**: Thresholds calibrated for 2-player games
   - **Future**: Adjust thresholds for 3-5 player variants

3. **Discard pile composition**: Not considered in classification
   - **Rationale**: Captured indirectly via progress (`c`) and deck size (`D`)
   - **Future**: Add critical card tracking (e.g., discarded 5s)

4. **Convention-specific strategies**: Assumes standard play patterns
   - **Future**: Encode advanced conventions (Finesse, Bluff) as pressure signals

### 7.2 Extensions

**Potential enhancements**:
1. **Sub-phases**: Early-early, early-late, etc. (finer granularity)
2. **Probability distributions**: `P(stage | s)` instead of deterministic labels
3. **Adaptive thresholds**: Learn optimal thresholds from win rate data
4. **Multi-dimensional visualization**: Plot state space in (t, c, pressure) coordinates

## 8. References

**Related work**:
- Hanabi is NP-hard (Baffier et al., 2017)
- Convention emergence in cooperative AI (Hu et al., 2020)
- Theory of mind in multi-agent systems (Rabinowitz et al., 2018)
