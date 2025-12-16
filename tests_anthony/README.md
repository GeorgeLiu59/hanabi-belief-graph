# Hanabi Belief Graph Unit Tests

## Purpose
Validate belief graph probability calculations in multi-agent Hanabi research.

## Critical Bugs Being Tested

### Bug #1: Uniform Probability Reset
- **What:** Cards on stacks get 20% probability instead of 0%
- **Evidence:** Turn 28 logs (lines 20590-20764)
- **Impact:** Agent plays impossible cards, loses lives
- **Test:** `test_belief_initialization.py::test_turn_28_white_stack_bug`
- **Root Cause:** `_create_belief_entry()` doesn't check fireworks state or discard pile

### Bug #2: Duplicate Card Misunderstanding
- **What:** Model doesn't understand multiple copies of same card exist
- **Evidence:** Process of elimination reasoning failures
- **Impact:** Incorrect probability calculations
- **Test:** TODO - not yet implemented

## Test Organization

### test_belief_initialization.py
Tests for belief creation and Turn 28 bug reproduction.
- Turn 28 bug scenario
- Individual impossible card tests
- Sanity checks for possible cards
- Belief entry structure validation

### test_belief_updates.py
Tests for belief updates across game progression.
- Multi-turn parametrized tests (turns 1, 5, 10, 15, 28)
- Update after PLAY actions
- Update after DISCARD actions
- Consistency across multiple plays

### test_input_formats.py
Tests for different input/output formats.
- Natural language belief descriptions
- JSON belief format validation
- Round-trip serialization
- Variant-specific formatting (certainty/probabilistic/ToM)

### test_prompt_consistency.py
Tests verifying production setup matching.
- Production prompt verification
- Model configuration verification
- Temperature setting validation
- CoT requirements presence

## Running Tests

### Run all tests
```bash
pytest tests_anthony/
```

### Run specific test file
```bash
pytest tests_anthony/test_belief_initialization.py
```

### Run specific test
```bash
pytest tests_anthony/test_belief_initialization.py::test_turn_28_white_stack_bug
```

### Run with coverage
```bash
pytest --cov=hanabi_learning_environment.agents tests_anthony/
```

### Run verbose
```bash
pytest -v tests_anthony/
```

### Collect tests without running
```bash
pytest --collect-only tests_anthony/
```

## Current Status

**Implementation Status:** Infrastructure complete, tests are skeletons

All test files have been created with:
- ✅ Proper module docstrings
- ✅ Detailed test function docstrings
- ✅ TODO comments for implementation
- ✅ `pytest.skip()` calls until implementation complete

**Next Steps:**
1. ✅ Extract production prompts to `tests_anthony/fixtures/production_prompts.txt` (COMPLETED)
2. Investigate agent API to understand:
   - How to create agents with specific game states
   - How to trigger belief creation
   - How to query belief probabilities
3. Implement actual test logic
4. Verify tests FAIL on current buggy code

## Expected Test Results

### Current Code (With Bugs)
Most tests should FAIL - this proves they catch the bugs.

**Expected failures:**
- `test_turn_28_white_stack_bug` - FAIL (P=0.2 instead of 0.0)
- `test_impossible_cards_at_each_turn` - FAIL at turn 28
- `test_white_1_zero_probability_when_white_stack_at_3` - FAIL

**Expected passes:**
- `test_production_prompt_matches_test_prompt` - PASS (after extraction)
- `test_belief_entry_creation_at_game_start` - PASS (bug only manifests mid-game)

### After Bug Fixes
All tests should PASS once bugs are fixed by Laksh/TQ.

## Adding New Tests

### When Laksh/TQ Find New Bugs

1. **Create fixture in `conftest.py`:**
```python
@pytest.fixture
def new_bug_scenario():
    return {
        "game_state": {...},
        "expected_behavior": {...}
    }
```

2. **Create test in appropriate file:**
```python
def test_new_bug(new_bug_scenario):
    """Document the bug clearly."""
    # Test implementation
    pass
```

3. **Document in README:**
   - Add to "Critical Bugs Being Tested" section
   - Reference log evidence
   - Link to test function

### Test Naming Convention
- `test_<specific_bug>` - Tests for known bugs
- `test_<component>_<behavior>` - General behavior tests
- Use descriptive names that explain what's being tested

## Fixtures Reference

### Available Fixtures (conftest.py)

- `hanabi_env`: Fresh 2-player Hanabi environment
- `probabilistic_agent`: Probabilistic belief graph agent instance
- `turn_28_game_state`: Exact Turn 28 bug scenario
- `turn_28_expected_beliefs`: Expected impossible/possible cards for Turn 28
- `turn_progression_scenarios`: Game states at turns 1, 5, 10, 15, 28

## Test Development Workflow

1. **Understand the bug** - Read game logs, identify failure point
2. **Create fixture** - Define game state that reproduces bug
3. **Write test skeleton** - Document expected behavior
4. **Investigate agent API** - Understand how to interact with code
5. **Implement test** - Write actual test logic
6. **Verify test fails** - Run on current buggy code, should FAIL
7. **Document** - Update README with test details

## Requirements

```bash
pip install pytest pytest-cov
```

## Project Structure

```
tests_anthony/
├── __init__.py              # Package init with overview
├── conftest.py              # Shared fixtures
├── test_belief_initialization.py
├── test_belief_updates.py
├── test_input_formats.py
├── test_prompt_consistency.py
├── README.md                # This file
└── fixtures/
    ├── game_states.json     # Pre-defined game scenarios
    └── production_prompts.txt  # Extracted production prompts

pytest.ini (root)            # Updated to point to tests_anthony/
```

## Links

- **Instructions:** `/Users/an/algoverse/hanabi1/instructions.md`
- **Agent Code:** `/Users/an/algoverse/hanabi1/hanabi_learning_environment/agents/`
- **Belief Graph Base:** `belief_graph_base.py`
- **Probabilistic Agent:** `probabilistic_agent.py`
- **Prompt Manager:** `prompt_manager.py`
