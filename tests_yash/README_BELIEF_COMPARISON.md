# WITH vs WITHOUT Belief Graph Testing

## Overview

The updated `agent_unit_testing_realistic.py` now supports testing agent decisions **WITH and WITHOUT belief graph scaffolding**. This allows you to validate whether the belief graph actually helps the agent make better decisions (not just by luck, but through better reasoning).

## Command-Line Usage

### Basic Commands

```bash
# Test all combinations (default) - 4 total runs
# (with_belief + new_cot, with_belief + old_cot, without_belief + new_cot, without_belief + old_cot)
python agent_unit_testing_realistic.py

# Test only WITH belief graph, new CoT prompt
python agent_unit_testing_realistic.py --belief-mode with --cot-mode new

# Test only WITHOUT belief graph, new CoT prompt
python agent_unit_testing_realistic.py --belief-mode without --cot-mode new

# Compare WITH vs WITHOUT using new CoT prompt only
python agent_unit_testing_realistic.py --belief-mode both --cot-mode new

# Test specific scenario file
python agent_unit_testing_realistic.py --scenarios scenarios/early_hint_scenarios.json --belief-mode both --cot-mode new
```

### Available Options

**`--belief-mode`**:
- `with` - Include belief graph sections (agent has access to probability distributions)
- `without` - Exclude belief graph sections (baseline agent with no memory/tracking)
- `both` - Test both modes for comparison (default)

**`--cot-mode`**:
- `new` - Use new risk-aware CoT prompt
- `old` - Use old biased CoT prompt
- `both` - Test both prompts (default)

**`--scenarios`**:
- Path to specific scenario JSON file
- If not provided, uses the RUNS list in the script

## Output Files

Files are saved with naming pattern: `{base_filename}_{belief_mode}_{cot_mode}.json`

Examples:
- `cot_outputs_jump_realistic2_with_belief_new_risk_aware.json`
- `cot_outputs_jump_realistic2_without_belief_new_risk_aware.json`
- `cot_outputs_jump_realistic2_with_belief_old_biased.json`
- `cot_outputs_jump_realistic2_without_belief_old_biased.json`

## What Changes Between WITH and WITHOUT?

### WITH Belief Graph
```
## BELIEF GRAPH ANALYSIS (PROBABILISTIC VARIANT)

**MY HAND BELIEFS:**
  - Card 1 (card_index 0):
    * Colors: GREEN(90%), RED(10%)
    * Ranks: 2(100%)
    * Status: 100% rank known (2), color uncertain

**TEAMMATE KNOWLEDGE MODEL:**
Player 2's cards:
  - Card 1: I see G2
    * Colors: GREEN(100%)
    * Ranks: 2(100%)
    * Status: Certain (GREEN 2)

## YOUR HAND
Card 1: Color: unknown, Rank: 2
Card 2: Color: unknown, Rank: unknown
...
```

### WITHOUT Belief Graph
```
## GAME STATE OBSERVATION (NO BELIEF TRACKING)

**IMPORTANT:** You do NOT have access to a belief graph system. You can only see:
- The current game state (fireworks, clues, lives, deck size)
- Your teammate's visible cards
- The history of what has been played/discarded
- What hints have been given this turn (if shown in history)

You CANNOT see your own cards, and you have NO MEMORY of prior hints given to you unless explicitly stated in the history.

## YOUR HAND
Card 1: Color: unknown, Rank: unknown
Card 2: Color: unknown, Rank: unknown
...
```

## Validation Approach

To validate belief graph effectiveness:

1. **Run both modes** on the same scenarios:
   ```bash
   python agent_unit_testing_realistic.py --belief-mode both --cot-mode new
   ```

2. **Compare agent reasoning** in the output files:
   - WITHOUT belief: Should miss important clues, give redundant hints, or fail to play certain cards
   - WITH belief: Should pick up on aggregated information, avoid redundancy, make optimal plays

3. **Check against scenario expectations**:
   - Each scenario has `with_belief_expected_hint` and `without_belief_redundant_hint`
   - Verify the agent's actual decision matches the expected behavior

## Example Comparison

**Scenario**: Turn 1, teammate has W1, G2, G4, B4, Y2

**WITHOUT belief graph**:
- Agent has no memory of teammate's cards being certain
- Likely gives color hint for visible cards (e.g., hint GREEN)
- Misses the optimal rank-1 hint that would enable immediate play

**WITH belief graph**:
- Agent tracks that W1 is the only rank-1
- Knows rank-1 hint has highest information value
- Correctly hints rank 1 to enable W1 play

## Tips

1. **Start small**: Test one scenario file with `--belief-mode both --cot-mode new`
2. **Compare outputs**: Look for reasoning quality differences
3. **Focus on new CoT**: The old biased prompt is mainly for historical comparison
4. **Use early_hint_scenarios.json**: Specifically designed for WITH/WITHOUT comparison
