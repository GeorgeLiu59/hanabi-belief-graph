# Hanabi Belief Graph Ablation Study

## Overview

This is a **research-focused ablation study** that scientifically tests whether belief graphs actually improve LLM agent decision-making in Hanabi, or whether simpler context representations work better.

## Research Question

**"Do belief graphs help or hurt agent reasoning?"**

Specifically:
1. Does the full probabilistic belief graph lead to better decisions than raw game state?
2. Does the Turn 28 bug (uniform 20% probabilities) make belief graphs counterproductive?
3. Does representation format (JSON vs natural language) matter more than content?
4. Would simpler contexts (minimal info, certainty-only) perform better?
5. Do agents actually USE belief information when provided?

## Methodology

### 5 Context Types Compared

1. **Full Belief Graph (JSON)** - Current system with complete probabilistic distributions
2. **Certainty Only** - Only 0%/100% probabilities (no uncertainty)
3. **Raw Game State** - Observable facts only (no belief representation)
4. **Minimal Context** - Bare minimum (just fireworks + hand size)
5. **Natural Language Belief** - Same beliefs as #1 but prose instead of JSON

### 7 Test Scenarios

All contexts tested on identical game states:
- Turn 1: Game start
- Turn 5: Minimal fireworks
- Turn 10: Medium complexity
- **Turn 28: White stack bug** (CRITICAL TEST)
- Turn 42: Late game
- Info tokens depleted
- Last life token

### Metrics Captured

For each decision:
- **Action safety**: SAFE / RISKY / FATAL
- **Matches ground truth**: Yes/No
- **Belief usage**: Did reasoning reference probabilities?
- **Reasoning quality**: Coherence score, logical flow
- **Bug susceptibility**: Did agent fall victim to Turn 28 bug?

## File Structure

```
tests_anthony/
├── ABLATION_STUDY_README.md          # This file
├── fixtures/
│   └── ablation_scenarios.json       # 7 test scenarios
├── context_generators.py             # Generate 5 context types
├── metrics.py                        # Data capture classes
├── analysis.py                       # Evaluation helper functions
├── reporting.py                      # Report generation
├── test_context_ablation.py          # Main ablation tests (35 tests)
├── test_reasoning_quality.py         # CoT analysis tests
├── conftest.py                       # pytest fixtures + session hooks
├── ablation_results.md               # Auto-generated report (after running tests)
└── ablation_results.json             # Raw data (after running tests)
```

## Running the Study

### Quick Start

```bash
# Run full ablation study
pytest tests_anthony/test_context_ablation.py -v

# View auto-generated report
cat tests_anthony/ablation_results.md
```

### Run Specific Tests

```bash
# Just Turn 28 bug scenario across all contexts
pytest tests_anthony/test_context_ablation.py::TestContextAblation::test_turn_28_white_stack_bug -v

# Just reasoning quality tests
pytest tests_anthony/test_reasoning_quality.py -v

# Specific context type
pytest tests_anthony/test_context_ablation.py -k "full_belief_graph" -v
```

### With Detailed Output

```bash
# See reasoning text for each decision
pytest tests_anthony/test_context_ablation.py -v -s
```

## How to Use

### Current Status

✅ **Infrastructure 100% Complete**
- 35 parametrized tests (7 scenarios × 5 contexts)
- Auto-report generation working
- Agent integration ready

### Step 1: Run with Mock Agent (Test Infrastructure)

```bash
# Quick test - no LLM calls, instant results
pytest tests_anthony/test_context_ablation.py -v

# View generated report
cat tests_anthony/ablation_results.md
```

This runs in ~0.03 seconds using ground truth actions.

### Step 2: Switch to Real Agent (For Research)

**Enable real agent mode:**

Edit `tests_anthony/conftest.py` line 191:
```python
USE_REAL_AGENT = True  # Change False to True
```

**Run the study:**
```bash
# Full study - 35 LLM calls
pytest tests_anthony/test_context_ablation.py -v

# Just Turn 28 scenario - 5 LLM calls (faster iteration)
pytest tests_anthony/test_context_ablation.py::TestContextAblation::test_turn_28_white_stack_bug -v
```

⚠️ **Note:** With real agent, this takes ~2-5 minutes and costs API credits.

### Step 3: Analyze Results

```bash
# View comprehensive markdown report
cat tests_anthony/ablation_results.md

# View raw JSON data
cat tests_anthony/ablation_results.json
```

### What You'll Learn

The auto-generated report answers your research questions:

**1. Which context performs best?**
```
Context Rankings (by safety):
1. Raw Game State: 85% safe ✅
2. Full Belief Graph: 57% safe
→ Conclusion: Simpler representation works better
```

**2. Does Turn 28 bug make belief graphs harmful?**
```
Full Belief Graph: PLAY White-1 (FATAL) ❌
Raw Game State: DISCARD White-1 (SAFE) ✅
→ Conclusion: Bug makes belief graphs counterproductive
```

**3. Do agents use belief information?**
```
Referenced probabilities: 2/7 times (29%)
→ Conclusion: Agents ignore most belief data
```

**4. Does format matter?**
```
JSON: 57% safe
Natural Language: 71% safe
→ Conclusion: Format matters more than content
```

These insights drive data-driven architecture decisions!

## Expected Findings

### Hypothesis 1: Belief Graphs Help
**Test**: Compare `full_belief_graph` vs `raw_game_state`
**If TRUE**: Full belief graph has higher safety percentage
**If FALSE**: Raw state performs equally well or better

### Hypothesis 2: Turn 28 Bug Makes Beliefs Harmful
**Test**: Turn 28 scenario specifically
**If TRUE**: Full belief graph → FATAL action, raw state → SAFE
**If FALSE**: Both lead to same action

### Hypothesis 3: Format Matters More Than Content
**Test**: `full_belief_graph` vs `natural_language_belief`
**If TRUE**: Different actions despite identical probabilities
**If FALSE**: Same action regardless of format

### Hypothesis 4: Simpler is Better
**Test**: `minimal_context` vs `full_belief_graph` reasoning quality
**If TRUE**: Minimal has higher coherence scores
**If FALSE**: Full context leads to better reasoning

### Hypothesis 5: Agents Don't Use Beliefs
**Test**: Check if reasoning references probabilities
**If TRUE**: Low belief usage even when provided
**If FALSE**: High belief usage when provided

## Output Reports

After running tests, two files are auto-generated:

### 1. ablation_results.md
Comprehensive markdown report with:
- Executive summary with context rankings
- Per-scenario analysis (which context won each scenario?)
- Per-context analysis (aggregate performance)
- Turn 28 bug deep dive
- Actionable recommendations

### 2. ablation_results.json
Raw data for further analysis:
```json
{
  "summary": {
    "total_tests_run": 35,
    "context_rankings_by_safety": [...],
    "context_performance": {...}
  },
  "detailed_results": [
    {
      "scenario_name": "turn_28_white_stack_bug",
      "context_type": "full_belief_graph",
      "action_type": "PLAY",
      "action_safety": "FATAL",
      "reasoning_text": "...",
      ...
    }
  ]
}
```

## Key Implementation Details

### Parametrized Testing
Each scenario runs across all 5 context types:
```python
@pytest.mark.parametrize("context_type", CONTEXT_TYPES)
class TestContextAblation:
    def test_turn_28_white_stack_bug(self, context_type, ...):
        # Same scenario, different context
        # 5 tests generated automatically
```

### Session-Level Reporting
pytest hook automatically generates report after all tests complete:
```python
def pytest_sessionfinish(session, exitstatus):
    # Auto-generate ablation_results.md
    # Auto-export ablation_results.json
    # Print summary statistics
```

### Safety Evaluation
Every action categorized as SAFE/RISKY/FATAL:
```python
def evaluate_action_safety(action, scenario):
    # Check if action in dangerous_actions list
    if matches_dangerous_action: return "FATAL"
    # Check if matches ground truth
    if matches_ground_truth: return "SAFE"
    # Otherwise suboptimal
    return "RISKY"
```

## Next Steps After Results

Based on findings, you can:

1. **If belief graphs help**:
   - Fix Turn 28 bug at `belief_graph_base.py:862`
   - Keep belief graph architecture

2. **If belief graphs don't help**:
   - Switch to best-performing representation
   - Simplify context (certainty-only, raw state)

3. **If format matters**:
   - Optimize representation (JSON vs natural language)
   - Redesign how beliefs are presented

4. **If agents ignore beliefs**:
   - Improve prompting to force belief engagement
   - Add explicit "use belief graph" instructions

## Scientific Value

This ablation study provides:
- **Scientific evidence** for/against belief graph architecture
- **Quantitative metrics** comparing different approaches
- **Reproducible results** with controlled scenarios
- **Publishable findings** for papers/presentations

Instead of guessing whether belief graphs help, you'll have data showing:
- Which context type leads to most safe decisions
- Whether the bug makes beliefs harmful
- How agents actually reason with different information
- What representation format works best

---

## Contact / Questions

This ablation study framework was designed to answer the research question: **"Are belief graphs actually worth it?"**

Run the study with your real agent, examine the results, and make data-driven architectural decisions!
