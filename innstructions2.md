# Hanabi Belief Graph Ablation Study - Testing Framework Implementation

## Research Objective

**Primary Question:** Do belief graphs actually improve LLM agent decision-making in Hanabi, or are there representation/reasoning issues that make them ineffective?

**Approach:** Ablation study comparing agent performance with different context representations using identical game scenarios.

---

## Background Context

### Current System
- Hanabi multi-agent game with cooperative card play
- Three belief graph variants: Certainty, Probabilistic, Theory of Mind
- GeminiAgent using Gemini 2.0 Flash Lite
- Known bug: Turn 28 - agent plays impossible cards due to uniform probability reset

### Research Hypothesis
Belief graphs should help agents make better decisions by explicitly representing card probabilities. However, we suspect:
1. The Turn 28 bug may make belief graphs ineffective
2. Representation format (JSON vs natural language) might matter more than content
3. Too much context could confuse rather than help
4. Agents might not actually use belief information in reasoning

---

## Test Design Philosophy

**NOT testing for bugs** (that's secondary)  
**TESTING for:** Does the belief graph representation improve agent reasoning compared to alternative context formats?

### Core Comparison Framework

Test the SAME game scenario with DIFFERENT context representations:

1. **Full Belief Graph** - Complete probabilistic distributions
2. **Certainty Only** - Only 0% or 100% probabilities (no uncertainty)
3. **Raw Game State** - Just observations (fireworks, discards, visible hands)
4. **Minimal Context** - Only current hand + fireworks stacks
5. **Natural Language Only** - Beliefs as prose instead of JSON

### What to Measure

For each context representation, capture:
- **Action chosen** (play/discard/hint + target)
- **Action safety** (safe, risky, or fatal)
- **Chain of Thought reasoning** (full text)
- **Belief graph usage** (did agent reference probabilities?)
- **Reasoning quality** (logical, uses available info, reaches correct conclusion)

---

## Implementation Tasks

### Task 1: Create Identical Test Scenarios

**File:** `tests_anthony/fixtures/ablation_scenarios.json`

Create 5-8 game scenarios where:
- Game state is identical across all tests
- Scenarios vary in complexity (early game, mid game, Turn 28 bug scenario)
- Each has a "ground truth" best action for comparison

**Scenario Structure:**
```json
{
  "scenario_name": "turn_28_white_stack_bug",
  "turn": 28,
  "fireworks": {"R": 3, "Y": 2, "G": 0, "W": 3, "B": 1},
  "life_tokens": 3,
  "info_tokens": 7,
  "current_player": 0,
  "player_hands": {
    "player_0": [
      {"color": "white", "rank": 1, "revealed": false},
      {"color": "blue", "rank": 4, "revealed": false}
    ],
    "player_1": [...]
  },
  "discard_pile": [...],
  "deck_size": 40,
  "ground_truth_best_action": {
    "type": "DISCARD",
    "target": 0,
    "reasoning": "White-1 is impossible (White stack at 3), safest to discard"
  },
  "dangerous_actions": [
    {"type": "PLAY", "target": 0, "consequence": "Loses life token"}
  ]
}
```

**Required Scenarios:**
1. Turn 1 (game start, no info)
2. Turn 5 (minimal fireworks, some hints given)
3. Turn 10 (medium complexity)
4. Turn 28 (the bug scenario - White stack at 3, agent has White-1)
5. Late game (Turn 40+, high stakes)
6. Info token depleted scenario
7. Last life token scenario

---

### Task 2: Context Generator Functions

**File:** `tests_anthony/context_generators.py`

Create functions that generate different context representations from the same game state.
```python
def generate_full_belief_graph_context(game_state: Dict) -> str:
    """
    Generate complete probabilistic belief graph as JSON.
    
    Example output:
    {
      "card_0": {
        "color_distribution": {"red": 0.2, "yellow": 0.2, "green": 0.2, "white": 0.2, "blue": 0.2},
        "rank_distribution": {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}
      },
      ...
    }
    """
    pass

def generate_certainty_only_context(game_state: Dict) -> str:
    """
    Only include certain knowledge (0% or 100% probabilities).
    
    Example:
    {
      "card_0": {
        "impossible": [("white", 1), ("white", 2), ("white", 3)],
        "certain": []
      }
    }
    """
    pass

def generate_raw_game_state_context(game_state: Dict) -> str:
    """
    Just observable facts, no belief representation.
    
    Example:
    "Fireworks: Red=3, Yellow=2, Green=0, White=3, Blue=1
     Your hand: [hidden], [hidden]
     Teammate's hand: Red-4, Blue-2
     Discard pile: White-1, White-2, ..."
    """
    pass

def generate_minimal_context(game_state: Dict) -> str:
    """
    Only current hand and fireworks.
    
    Example:
    "Fireworks: R=3, Y=2, G=0, W=3, B=1
     Your hand: 2 cards"
    """
    pass

def generate_natural_language_belief_context(game_state: Dict) -> str:
    """
    Belief graph as prose instead of JSON.
    
    Example:
    "For your first card: There's a 20% chance it's white, 20% chance it's red...
     Based on what's been played, it definitely cannot be white-1, white-2, or white-3."
    """
    pass
```

---

### Task 3: Core Ablation Test

**File:** `tests_anthony/test_context_ablation.py`
```python
import pytest
from typing import Dict, Tuple

@pytest.fixture
def ablation_scenarios():
    """Load pre-defined identical scenarios."""
    import json
    with open('tests_anthony/fixtures/ablation_scenarios.json') as f:
        return json.load(f)

@pytest.mark.parametrize("context_type", [
    "full_belief_graph",
    "certainty_only",
    "raw_game_state",
    "minimal_context",
    "natural_language_belief"
])
class TestContextAblation:
    """Compare agent decisions across different context representations."""
    
    def test_turn_28_scenario_across_contexts(self, context_type, ablation_scenarios):
        """
        Turn 28 bug scenario - most critical test.
        
        Expected insights:
        - Does full belief graph prevent bad action? (probably NO due to bug)
        - Does raw game state lead to better decision? (possibly YES)
        - Does agent use belief information at all?
        """
        scenario = ablation_scenarios["turn_28_white_stack_bug"]
        
        # Generate context based on type
        context = generate_context(scenario, context_type)
        
        # Get agent decision + reasoning
        action, reasoning = agent.decide_with_reasoning(scenario, context)
        
        # Evaluate decision quality
        safety = evaluate_action_safety(action, scenario)
        
        # Analyze reasoning
        belief_usage = analyze_belief_graph_usage(reasoning, context_type)
        
        # Log results for comparison
        log_ablation_result({
            "scenario": "turn_28",
            "context_type": context_type,
            "action": action,
            "safety": safety,
            "reasoning": reasoning,
            "used_beliefs": belief_usage,
            "ground_truth_action": scenario["ground_truth_best_action"]
        })
        
        # Assertion: Document whether this context led to safe decision
        assert_and_document(
            actual=action,
            expected=scenario["ground_truth_best_action"],
            context=context_type,
            message=f"{context_type} context led to {safety} action"
        )
    
    def test_early_game_scenario_across_contexts(self, context_type, ablation_scenarios):
        """Test turn 1 scenario - minimal information available."""
        # Similar structure to above
        pass
    
    def test_late_game_scenario_across_contexts(self, context_type, ablation_scenarios):
        """Test high-stakes late game scenario."""
        pass
```

---

### Task 4: Chain of Thought Analysis

**File:** `tests_anthony/test_reasoning_quality.py`
```python
class TestReasoningQuality:
    """Analyze HOW agent reasons with different contexts."""
    
    def test_belief_graph_referenced_in_reasoning(self):
        """
        Question: Does agent actually USE belief graph when given one?
        
        Check if CoT mentions:
        - Probabilities ("20% chance", "unlikely", etc.)
        - Impossible cards ("cannot be white-1")
        - Distributions ("most likely red or blue")
        """
        scenario = load_scenario("turn_28")
        context = generate_full_belief_graph_context(scenario)
        
        action, reasoning = agent.decide(scenario, context)
        
        # Check for probability language
        assert any(word in reasoning.lower() for word in ["probability", "chance", "likely", "%"]), \
            "Agent should reference probabilities when given belief graph"
        
        # Check for impossibility reasoning
        assert any(phrase in reasoning.lower() for phrase in ["cannot be", "impossible", "ruled out"]), \
            "Agent should reason about impossible cards"
        
        # Log full reasoning for manual analysis
        print(f"\n=== REASONING WITH BELIEF GRAPH ===\n{reasoning}\n")
    
    def test_reasoning_without_belief_graph(self):
        """
        Question: How does agent reason without belief graph?
        
        Hypothesis: Might be clearer/simpler reasoning.
        """
        scenario = load_scenario("turn_28")
        context = generate_raw_game_state_context(scenario)
        
        action, reasoning = agent.decide(scenario, context)
        
        # Check what agent bases reasoning on
        assert "stack" in reasoning.lower() or "fireworks" in reasoning.lower(), \
            "Agent should reason about observable game state"
        
        print(f"\n=== REASONING WITHOUT BELIEF GRAPH ===\n{reasoning}\n")
    
    def test_reasoning_quality_comparison(self):
        """
        Compare reasoning quality across contexts.
        
        Metrics:
        - Logical coherence
        - Appropriate use of available information
        - Correct conclusion from premises
        """
        scenario = load_scenario("turn_28")
        
        results = {}
        for context_type in CONTEXT_TYPES:
            context = generate_context(scenario, context_type)
            action, reasoning = agent.decide(scenario, context)
            
            results[context_type] = {
                "reasoning_length": len(reasoning.split()),
                "mentions_fireworks": "stack" in reasoning.lower(),
                "mentions_probabilities": "%" in reasoning or "chance" in reasoning.lower(),
                "logical_flow": check_logical_coherence(reasoning),
                "action_matches_reasoning": check_consistency(action, reasoning)
            }
        
        # Generate comparison report
        generate_reasoning_comparison_report(results)
```

---

### Task 5: Metrics Capture System

**File:** `tests_anthony/metrics.py`
```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class AgentDecisionMetrics:
    """Captures all relevant metrics for a single agent decision."""
    
    scenario_name: str
    context_type: str
    turn: int
    
    # Decision outputs
    action_type: str  # "PLAY", "DISCARD", "HINT"
    action_target: int
    reasoning_text: str
    
    # Safety evaluation
    action_safety: str  # "SAFE", "RISKY", "FATAL"
    would_lose_life: bool
    is_optimal: bool
    matches_ground_truth: bool
    
    # Belief graph usage
    referenced_probabilities: bool
    mentioned_impossible_cards: bool
    used_numerical_probs: bool  # "20%" vs "unlikely"
    probability_interpretation_correct: bool
    
    # Reasoning quality
    reasoning_length_words: int
    references_fireworks: bool
    references_discards: bool
    references_teammate_hand: bool
    logical_coherence_score: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return self.__dict__
    
    @classmethod
    def from_agent_output(cls, scenario, context_type, action, reasoning):
        """Factory method to create metrics from agent outputs."""
        return cls(
            scenario_name=scenario["scenario_name"],
            context_type=context_type,
            turn=scenario["turn"],
            action_type=action["type"],
            action_target=action["target"],
            reasoning_text=reasoning,
            # ... compute all other metrics
        )

class MetricsCollector:
    """Collects and aggregates metrics across all tests."""
    
    def __init__(self):
        self.results: List[AgentDecisionMetrics] = []
    
    def add_result(self, metrics: AgentDecisionMetrics):
        """Add a single test result."""
        self.results.append(metrics)
    
    def generate_comparison_report(self) -> str:
        """
        Generate comprehensive comparison across context types.
        
        Returns markdown report showing:
        - Which context led to best decisions
        - Belief graph usage patterns
        - Reasoning quality differences
        """
        pass
    
    def generate_per_scenario_report(self, scenario_name: str) -> str:
        """
        Focus on one scenario across all context types.
        
        Example: "Turn 28 scenario - how did each context perform?"
        """
        pass
    
    def export_to_json(self, filepath: str):
        """Export all results for further analysis."""
        pass
```

---

### Task 6: Analysis Helper Functions

**File:** `tests_anthony/analysis.py`
```python
def evaluate_action_safety(action: Dict, scenario: Dict) -> str:
    """
    Determine if action is SAFE, RISKY, or FATAL.
    
    SAFE: No negative consequences
    RISKY: Suboptimal but not dangerous
    FATAL: Will lose life token or game
    """
    dangerous_actions = scenario.get("dangerous_actions", [])
    
    for dangerous in dangerous_actions:
        if action["type"] == dangerous["type"] and action["target"] == dangerous["target"]:
            return "FATAL"
    
    if action != scenario["ground_truth_best_action"]:
        return "RISKY"
    
    return "SAFE"

def analyze_belief_graph_usage(reasoning: str, context_type: str) -> Dict:
    """
    Analyze whether and how agent used belief graph information.
    
    Returns metrics about belief usage in reasoning.
    """
    if context_type not in ["full_belief_graph", "natural_language_belief"]:
        return {"applicable": False}
    
    return {
        "applicable": True,
        "mentioned_probabilities": any(word in reasoning.lower() for word in 
            ["probability", "chance", "likely", "%", "distribution"]),
        "mentioned_impossible": any(phrase in reasoning.lower() for phrase in
            ["cannot be", "impossible", "ruled out", "definitely not"]),
        "used_numerical_values": bool(re.search(r'\d+%', reasoning)),
        "probability_count": reasoning.lower().count("probability") + reasoning.count("%"),
        "reasoning_quality": "high" if "cannot be" in reasoning.lower() else "low"
    }

def check_logical_coherence(reasoning: str) -> float:
    """
    Score reasoning logical coherence from 0.0 to 1.0.
    
    Checks:
    - Premises stated
    - Logical flow (if X then Y)
    - Conclusion matches premises
    """
    score = 0.0
    
    # Has premises?
    if any(word in reasoning.lower() for word in ["because", "since", "given that"]):
        score += 0.3
    
    # Has logical connectors?
    if any(word in reasoning.lower() for word in ["therefore", "so", "thus", "hence"]):
        score += 0.3
    
    # Has conclusion?
    if any(phrase in reasoning.lower() for phrase in ["i will", "i should", "best to"]):
        score += 0.2
    
    # References game state?
    if any(word in reasoning.lower() for word in ["stack", "firework", "discard", "hand"]):
        score += 0.2
    
    return min(score, 1.0)

def check_consistency(action: Dict, reasoning: str) -> bool:
    """
    Check if action matches what reasoning suggests.
    
    Example: Reasoning says "I should discard" but action is PLAY → False
    """
    action_mentioned = action["type"].lower() in reasoning.lower()
    
    # Check for contradictions
    if action["type"] == "PLAY" and "discard" in reasoning.lower() and "play" not in reasoning.lower():
        return False
    
    if action["type"] == "DISCARD" and "play" in reasoning.lower() and "discard" not in reasoning.lower():
        return False
    
    return action_mentioned
```

---

### Task 7: Reporting and Visualization

**File:** `tests_anthony/reporting.py`
```python
def generate_ablation_report(results: List[AgentDecisionMetrics]) -> str:
    """
    Generate comprehensive markdown report.
    
    Sections:
    1. Executive Summary
    2. Per-Scenario Analysis
    3. Per-Context Analysis
    4. Belief Graph Usage Findings
    5. Recommendations
    """
    report = """
# Hanabi Belief Graph Ablation Study - Results

## Executive Summary

### Key Findings
- [AUTO-GENERATED based on results]

### Context Performance Ranking
1. [Best performing context]
2. [Second best]
...

---

## Scenario Analysis

### Turn 28 (White Stack Bug)

**Ground Truth Best Action:** DISCARD card 0

| Context Type | Action | Safety | Used Beliefs | Reasoning Quality |
|--------------|--------|--------|--------------|-------------------|
| Full Belief Graph | PLAY card 0 | FATAL ❌ | Yes (incorrectly) | 6/10 |
| Certainty Only | DISCARD card 0 | SAFE ✅ | Yes (correctly) | 8/10 |
| Raw Game State | DISCARD card 0 | SAFE ✅ | N/A | 7/10 |
| Minimal Context | HINT color | RISKY ⚠️ | N/A | 5/10 |
| Natural Language | PLAY card 0 | FATAL ❌ | Yes (incorrectly) | 6/10 |

**Analysis:**
Full belief graph led to WORSE decision due to Turn 28 bug (20% probability for impossible card).
Certainty-only and raw game state both led to safe decisions.
Agent WITH belief graph made fatal error, agent WITHOUT made safe choice.

**Sample Reasoning (Full Belief Graph):**
"I have a 20% chance this card is white-1. The white stack is at 3, but there's still a chance..."

**Sample Reasoning (Raw Game State):**
"White stack is at 3, meaning white-1, white-2, white-3 already played. My card cannot be any of these. Safest to discard."

---

[Repeat for each scenario]

---

## Context Type Analysis

### Full Belief Graph

**Safe Decisions:** 2/7 scenarios (29%)
**Fatal Decisions:** 3/7 scenarios (43%)

**Belief Usage:**
- Referenced probabilities: 6/7 times (86%)
- Correctly interpreted probabilities: 2/7 times (29%)
- Mentioned impossible cards: 4/7 times (57%)

**Insight:** Agent uses belief graph but often misinterprets probabilities, especially when bug causes incorrect values.

---

[Repeat for each context type]

---

## Recommendations

1. **Fix Turn 28 Bug First**
   - Current belief graph is counterproductive
   - Agent makes worse decisions WITH belief graph than without

2. **Consider Certainty-Only Representation**
   - Performed better than probabilistic in X/7 scenarios
   - Simpler for LLM to reason about

3. **Natural Language May Be Better Than JSON**
   - Numerical probabilities confuse agent
   - Qualitative descriptions ("likely", "impossible") work better

4. **Minimal Context Surprisingly Effective**
   - Agent forced to use own reasoning
   - Led to safe decisions in Y/7 scenarios

"""
    return report

def export_results_csv(results: List[AgentDecisionMetrics], filepath: str):
    """Export results as CSV for further analysis in Excel/Python."""
    pass

def create_comparison_visualization(results: List[AgentDecisionMetrics]):
    """
    Create visual comparison charts (if matplotlib available).
    
    Charts:
    - Safety by context type (bar chart)
    - Reasoning quality by context (box plot)
    - Belief usage patterns (heatmap)
    """
    pass
```

---

### Task 8: Pytest Integration

**File:** `tests_anthony/conftest.py` (additions)
```python
import pytest
import json
from pathlib import Path

# Metrics collector shared across all tests
_metrics_collector = MetricsCollector()

@pytest.fixture(scope="session")
def metrics_collector():
    """Shared metrics collector for all ablation tests."""
    return _metrics_collector

@pytest.fixture
def ablation_scenarios():
    """Load all ablation test scenarios."""
    filepath = Path(__file__).parent / "fixtures" / "ablation_scenarios.json"
    with open(filepath) as f:
        return json.load(f)

@pytest.fixture
def agent():
    """Instantiate agent for testing."""
    # TODO: Configure agent properly
    return BeliefGraphProbabilisticAgent(config={"players": 2})

def pytest_sessionfinish(session, exitstatus):
    """
    After all tests complete, generate comparison report.
    
    This runs automatically at the end of pytest session.
    """
    report = _metrics_collector.generate_comparison_report()
    
    # Save report
    report_path = Path(__file__).parent / "ablation_results.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print(f"Ablation study complete! Report saved to:")
    print(f"  {report_path}")
    print(f"{'='*60}\n")
    
    # Also export raw data
    _metrics_collector.export_to_json(
        Path(__file__).parent / "ablation_results.json"
    )
```

---

## Expected Insights from Ablation Study

### Hypothesis 1: Belief Graphs Help
**Test:** Compare full_belief_graph vs raw_game_state  
**Expected if TRUE:** Full belief graph leads to more safe decisions  
**Expected if FALSE:** Raw game state performs equally well or better

### Hypothesis 2: Turn 28 Bug Makes Belief Graphs Harmful
**Test:** Turn 28 scenario specifically  
**Expected if TRUE:** Full belief graph leads to FATAL action, raw state leads to SAFE  
**Expected if FALSE:** Both lead to same action

### Hypothesis 3: Format Matters More Than Content
**Test:** full_belief_graph vs natural_language_belief (same probabilities, different format)  
**Expected if TRUE:** Different actions despite identical information  
**Expected if FALSE:** Same action regardless of format

### Hypothesis 4: Simpler is Better
**Test:** Compare reasoning quality: minimal_context vs full_belief_graph  
**Expected if TRUE:** Minimal context has higher coherence scores  
**Expected if FALSE:** Full context leads to better reasoning

### Hypothesis 5: Agent Doesn't Actually Use Beliefs
**Test:** Check if reasoning references probabilities  
**Expected if TRUE:** Low belief usage even when provided  
**Expected if FALSE:** High belief usage when provided

---

## Deliverables Checklist

After implementing this framework:

- [ ] `ablation_scenarios.json` with 7+ scenarios
- [ ] `context_generators.py` with 5 generation functions
- [ ] `test_context_ablation.py` with parametrized tests
- [ ] `test_reasoning_quality.py` with CoT analysis
- [ ] `metrics.py` with data capture classes
- [ ] `analysis.py` with evaluation functions
- [ ] `reporting.py` with report generation
- [ ] Updated `conftest.py` with session-level reporting
- [ ] Tests run successfully and generate report
- [ ] `ablation_results.md` report file created
- [ ] `ablation_results.json` raw data exported

---

## Running the Ablation Study
```bash
# Run all ablation tests
pytest tests_anthony/test_context_ablation.py -v

# Run specific scenario across all contexts
pytest tests_anthony/test_context_ablation.py::TestContextAblation::test_turn_28_scenario_across_contexts -v

# Run with detailed output
pytest tests_anthony/test_context_ablation.py -v -s

# After completion, view report
cat tests_anthony/ablation_results.md
```

---

## Success Criteria

This implementation is successful if:

1. ✅ Same scenarios tested across all 5 context types
2. ✅ Agent decisions and reasoning captured for each
3. ✅ Metrics computed automatically
4. ✅ Comparison report generated showing which context performs best
5. ✅ Clear evidence for/against belief graph effectiveness
6. ✅ Insights exportable for paper/presentation

**Ultimate Goal:** Determine scientifically whether belief graphs improve agent reasoning, or if simpler representations work better.

---

## Next Steps After Results

Based on findings, you might:
- **If belief graphs help:** Focus on fixing Turn 28 bug, keep belief graph approach
- **If belief graphs don't help:** Consider switching to simpler representation (certainty-only, raw state)
- **If format matters:** Optimize how beliefs are presented to LLM (natural language > JSON)
- **If agent ignores beliefs:** Redesign prompts to force belief engagement

This ablation study will give you scientific evidence for architectural decisions!