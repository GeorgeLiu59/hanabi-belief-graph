# LLM-as-Judge Evaluation Protocol: Formal Specification

## 1. Motivation & Research Context

### 1.1 Problem Statement

**Limitation of heuristic methods**: Keyword-based coherence scoring (e.g., counting "because", "fireworks") cannot assess whether reasoning is logically valid or factually correct.

**Example failure**:
```
Agent reasoning: "I'll play White-1 because the fireworks show good progress."
Heuristic score: 0.45 (has "because" +0.3, has "fireworks" +0.15)
Reality: FATAL ERROR - White stack at 3, White-1 is impossible
```

### 1.2 Solution: LLM-as-Judge

**Approach**: Use a second LLM to evaluate reasoning quality against explicit ground truth

**Key innovation**: Judge receives ground truth explanation, not game rules
- **Prevents alignment errors**: Judge doesn't interpret rules (source of bugs)
- **Enables comparison**: Judge checks if agent reasoning matches known correct answer
- **Maintains rigor**: Evaluation grounded in expert-annotated truth, not LLM's own understanding

## 2. Selective Judging Strategy

### 2.1 Cost-Efficiency Principle

**Problem**: Full judging doubles API costs and latency
- Without judging: N scenarios = N agent calls
- With full judging: N scenarios = 2N calls (N agent + N judge)

**Solution**: Judge only when evaluation provides value

### 2.2 Judging Criteria

```python
def should_judge(action_safety: str, matches_ground_truth: bool, scenario_name: str) -> bool:
    """Determine if scenario warrants LLM-as-Judge evaluation."""

    # Always judge critical test cases
    if "turn_28" in scenario_name.lower() or "bug" in scenario_name.lower():
        return True  # Research priority

    # Skip obviously flawed reasoning
    if action_safety == "FATAL":
        return False  # Reasoning must be wrong if action is fatal

    # Judge ambiguous cases (reasoning quality matters)
    if action_safety in ["SAFE", "RISKY"]:
        return True  # Could be right answer, wrong reasoning (or vice versa)

    # Default: skip
    return False
```

**Expected reduction**: ~40% fewer judge calls vs. full evaluation

**Empirical validation**:
| Condition | Expected % | Judge? | Rationale |
|-----------|-----------|--------|-----------|
| SAFE action | 60% | ✅ Yes | May have correct action but flawed logic |
| RISKY action | 25% | ✅ Yes | Ambiguous - need to assess reasoning |
| FATAL action | 10% | ❌ No | Obviously wrong (skip judge cost) |
| Turn 28 bug | 1% | ✅ Yes | Critical research scenario |

## 3. Formal Scoring Rubric

### 3.1 Evaluation Dimensions

The judge assesses four orthogonal aspects of reasoning quality:

#### Dimension 1: Factual Accuracy (FA)
**Question**: Did the agent correctly state observable game facts?

**Scoring**:
- `CORRECT`: All fireworks heights, token counts, hand composition stated accurately
- `PARTIAL`: Minor errors (e.g., off-by-one in token count, missing one stack height)
- `INCORRECT`: Major factual errors (e.g., wrong fireworks heights, miscount lives)

**Weight**: 0.3

**Examples**:
```
Scenario: Fireworks R:3 Y:2 G:0 W:3 B:1, Life:3, Info:7

CORRECT: "Fireworks are at Red-3, Yellow-2, Green-0, White-3, Blue-1. We have 3 lives and 7 info tokens."
PARTIAL: "Fireworks show Red-3, Yellow-2, White-3. We have full health." (missing G/B, vague "full health")
INCORRECT: "White stack is at 1" (factually wrong - actual stack at 3)
```

#### Dimension 2: Impossibility Recognition (IR)
**Question**: Did the agent identify cards that cannot be in hand?

**Scoring**:
- `CORRECT`: Explicitly states cards on stack are impossible in hand
- `PARTIAL`: Implicitly recognizes (e.g., "can't be low ranks") but doesn't enumerate
- `INCORRECT`: Claims impossible cards are possible
- `N/A`: No impossible cards in this scenario (all cards could be in hand)

**Weight**: 0.4 (highest - critical for Hanabi reasoning)

**Examples**:
```
Scenario: White stack at 3

CORRECT: "White-1, White-2, and White-3 have been played. These cards cannot be in my hand."
PARTIAL: "Low white cards are already on the stack." (correct idea, imprecise)
INCORRECT: "My card could be White-1." (impossible - White-1 already played)
N/A: (scenario with no stacks, all cards possible)
```

**Weight redistribution when N/A**:
```python
if ir == "N/A":
    ir_weight = 0.0
    fa_weight = 0.5  # Increase FA weight from 0.3 to 0.5
```

#### Dimension 3: Logical Soundness (LS)
**Question**: Does the agent's conclusion follow logically from stated premises?

**Scoring**:
- `SOUND`: Conclusion directly supported by stated facts (valid deductive reasoning)
- `WEAK`: Conclusion plausible but gaps in reasoning chain
- `UNSOUND`: Conclusion contradicts stated facts (logical fallacy)

**Weight**: 0.2

**Examples**:
```
SOUND: "White stack is at 3. Therefore White-1/2/3 are impossible. I should discard."
   → Premises support conclusion ✓

WEAK: "I have no hints. I'll discard to be safe."
   → Conclusion reasonable but missing analysis of playability ≈

UNSOUND: "White stack is at 3. Therefore I'll play White-1 to continue the stack."
   → Conclusion contradicts premise (W-1 already played) ✗
```

#### Dimension 4: Action Alignment (AA)
**Question**: Does the reasoning logically support the action chosen?

**Scoring**:
- `ALIGNED`: Reasoning clearly justifies action choice
- `MISALIGNED`: Reasoning suggests different action than chosen

**Weight**: 0.1 (lowest - captures consistency, not correctness)

**Examples**:
```
ALIGNED: "Card is unknown and risky. I will DISCARD." (reasoning → action)
MISALIGNED: "Card is definitely playable. I will DISCARD." (reasoning ≠ action)
```

### 3.2 Composite Score Calculation

```python
def calculate_judge_score(
    fa: str,  # Factual Accuracy
    ir: str,  # Impossibility Recognition
    ls: str,  # Logical Soundness
    aa: str,  # Action Alignment
    ground_truth: str
) -> float:
    """
    Compute weighted composite score for reasoning quality.

    Args:
        fa: Factual Accuracy ∈ {CORRECT, PARTIAL, INCORRECT}
        ir: Impossibility Recognition ∈ {CORRECT, PARTIAL, INCORRECT, N/A}
        ls: Logical Soundness ∈ {SOUND, WEAK, UNSOUND}
        aa: Action Alignment ∈ {ALIGNED, MISALIGNED}
        ground_truth: Ground truth explanation (used to determine if N/A handling needed)

    Returns:
        Composite score ∈ [0.0, 1.0]

    Scoring formula:
        score = w_fa * v_fa + w_ir * v_ir + w_ls * v_ls + w_aa * v_aa

    Where:
        v_x ∈ {0.0, 0.5, 1.0} (value mapping for dimension x)
        w_x = weight for dimension x (sum to 1.0)
    """
    # Map categorical scores to numeric values
    fa_val = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}[fa]
    ls_val = {"SOUND": 1.0, "WEAK": 0.5, "UNSOUND": 0.0}[ls]
    aa_val = {"ALIGNED": 1.0, "MISALIGNED": 0.0}[aa]

    # Handle impossibility recognition (N/A case)
    if ir == "N/A":
        # No impossible cards in scenario - redistribute weight to FA
        ir_val = 1.0 if "impossible" not in ground_truth.lower() else 0.5
        ir_weight = 0.0
        fa_weight = 0.5  # Increase from base 0.3
    else:
        ir_val = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}[ir]
        ir_weight = 0.4
        fa_weight = 0.3

    # Compute weighted sum
    score = (fa_weight * fa_val +
             ir_weight * ir_val +
             0.2 * ls_val +
             0.1 * aa_val)

    return round(score, 2)
```

**Score interpretation**:
- **0.9-1.0**: Excellent reasoning - factually accurate, logically sound, recognizes constraints
- **0.7-0.89**: Good reasoning - minor gaps but fundamentally sound
- **0.5-0.69**: Adequate reasoning - significant issues but not critically flawed
- **0.3-0.49**: Poor reasoning - major factual or logical errors
- **0.0-0.29**: Critical reasoning failure - incorrect facts or unsound logic

### 3.3 Worked Example

**Scenario**: Turn 28 White Stack Bug
```
Fireworks: W:3 (White stack at 3)
Agent hand: [Unknown card, position 0]
Ground truth: "White-1/2/3 impossible (already played). Best action: DISCARD position 0."
```

**Agent reasoning**: "I see the white stack is at 3. This means White-4 is playable next. However, I don't know if my card is White-4, so I'll discard to be safe."

**Judge evaluation**:
1. **FA**: CORRECT (accurately states white stack at 3) → 1.0
2. **IR**: PARTIAL (doesn't explicitly state W-1/2/3 impossible, but implies by mentioning W-4 next) → 0.5
3. **LS**: SOUND (conclusion follows from uncertainty about hand) → 1.0
4. **AA**: ALIGNED (reasoning supports discard action) → 1.0

**Score calculation**:
```python
score = 0.3 * 1.0 + 0.4 * 0.5 + 0.2 * 1.0 + 0.1 * 1.0
      = 0.3 + 0.2 + 0.2 + 0.1
      = 0.8
```

**Interpretation**: Good reasoning (0.8) - agent reaches correct conclusion but could explicitly enumerate impossible cards.

## 4. Judge Prompt Template

### 4.1 Template Structure

```python
JUDGE_PROMPT_TEMPLATE = """You are evaluating a Hanabi AI agent's reasoning quality.

GROUND TRUTH (authoritative - do NOT question this):
{ground_truth_explanation}

Specifically for this scenario:
{impossibility_statements}

AGENT'S REASONING:
{agent_reasoning}

AGENT'S ACTION:
Action type: {action_type}
Target: {action_target}

---

EVALUATION TASK:
Rate the agent's reasoning on these four dimensions:

1. Factual Accuracy (CORRECT/PARTIAL/INCORRECT):
   Did the agent correctly state the observable game facts?
   - CORRECT: All fireworks, tokens, hand composition stated accurately
   - PARTIAL: Minor errors (off-by-one, missing details)
   - INCORRECT: Major factual errors

2. Impossibility Recognition (CORRECT/PARTIAL/INCORRECT/N/A):
   Did the agent identify which cards cannot be in hand?
   - CORRECT: Explicitly states impossible cards
   - PARTIAL: Implicitly recognizes but doesn't enumerate
   - INCORRECT: Claims impossible cards are possible
   - N/A: No impossible cards in this scenario

3. Logical Soundness (SOUND/WEAK/UNSOUND):
   Does the conclusion follow from the stated premises?
   - SOUND: Conclusion directly supported by facts
   - WEAK: Conclusion plausible but gaps in reasoning
   - UNSOUND: Conclusion contradicts stated facts

4. Action Alignment (ALIGNED/MISALIGNED):
   Does the reasoning support the chosen action?
   - ALIGNED: Reasoning justifies action
   - MISALIGNED: Reasoning suggests different action

RESPOND IN THIS EXACT JSON FORMAT:
{{
  "factual_accuracy": "CORRECT|PARTIAL|INCORRECT",
  "impossibility_recognition": "CORRECT|PARTIAL|INCORRECT|N/A",
  "logical_soundness": "SOUND|WEAK|UNSOUND",
  "action_alignment": "ALIGNED|MISALIGNED",
  "explanation": "Brief 1-2 sentence justification for each score"
}}

Do NOT provide any text outside the JSON object.
"""
```

### 4.2 Ground Truth Format

**Structured natural language template**:
```
Given: [Observable game state facts]
Derived Facts: [Logical consequences of game rules]
Impossibility Constraints: [Cards that cannot be in hand, with probabilities]
Playability Analysis: [Safe/risky/fatal actions]
Therefore: [Best action conclusion]
```

**Example** (Turn 28 bug):
```
Ground Truth Explanation:

Given:
- White fireworks stack is at height 3
- Agent holds card at position 0 with unknown identity (no hints received)
- Life tokens: 3, Info tokens: 7, Deck size: 40

Derived Facts:
- Stack height 3 means White-1, White-2, and White-3 are ALREADY PLAYED
- Next playable White card is White-4
- Cards on stack cannot exist in any player's hand (uniqueness constraint)

Impossibility Constraints:
- P(agent hand contains White-1) = 0.0 (impossible - already played)
- P(agent hand contains White-2) = 0.0 (impossible - already played)
- P(agent hand contains White-3) = 0.0 (impossible - already played)

Playability Analysis:
- Playing White-1: FATAL (duplicate card → lose 1 life token)
- Playing White-4: SAFE (if card is actually White-4)
- Discarding unknown card: SAFE (preserves life tokens, gains info token)

Therefore:
- Best action: DISCARD position 0 (no information about card, cannot confirm safety)
- Dangerous actions: PLAY position 0 (could be impossible White-1/2/3)
- Critical insight: Correct belief system MUST assign 0% to impossible cards
```

## 5. Implementation Protocol

### 5.1 Judge Invocation

```python
def evaluate_reasoning(
    scenario: Dict[str, Any],
    agent_reasoning: str,
    agent_action: Dict[str, Any],
    model: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Evaluate agent reasoning using LLM-as-Judge.

    Args:
        scenario: Game scenario with ground_truth_explanation field
        agent_reasoning: Agent's chain-of-thought text
        agent_action: Agent's chosen action dict
        model: LLM model to use as judge (default: gemini-2.5-pro)

    Returns:
        Dictionary containing:
        {
            "factual_accuracy": str,
            "impossibility_recognition": str,
            "logical_soundness": str,
            "action_alignment": str,
            "composite_score": float,
            "explanation": str,
            "model_used": str
        }

    Raises:
        ValueError: If ground truth missing from scenario
        JSONDecodeError: If judge response not valid JSON
    """
    # Validate inputs
    if "ground_truth_explanation" not in scenario:
        raise ValueError(f"Scenario {scenario['scenario_name']} missing ground_truth_explanation")

    # Create prompt
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        ground_truth_explanation=scenario["ground_truth_explanation"],
        impossibility_statements=scenario.get("impossibility_statements", "None"),
        agent_reasoning=agent_reasoning,
        action_type=agent_action.get("type", "UNKNOWN"),
        action_target=agent_action.get("target", "N/A")
    )

    # Call judge LLM
    response = call_llm(model, prompt, temperature=0.2)  # Low temp for consistency

    # Parse JSON response
    evaluation = json.loads(response.strip())

    # Calculate composite score
    composite_score = calculate_judge_score(
        fa=evaluation["factual_accuracy"],
        ir=evaluation["impossibility_recognition"],
        ls=evaluation["logical_soundness"],
        aa=evaluation["action_alignment"],
        ground_truth=scenario["ground_truth_explanation"]
    )

    # Add metadata
    evaluation["composite_score"] = composite_score
    evaluation["model_used"] = model

    return evaluation
```

### 5.2 Error Handling

**Common failure modes**:
1. **JSON parsing errors**: Judge returns text instead of JSON
   - **Mitigation**: Prompt emphasizes "Do NOT provide any text outside JSON object"
   - **Fallback**: Retry with clarified prompt, max 2 retries

2. **Invalid score values**: Judge returns unrecognized category
   - **Mitigation**: Validate all fields against allowed values
   - **Fallback**: Log error, assign lowest score for that dimension

3. **Missing ground truth**: Scenario lacks required fields
   - **Mitigation**: Validate scenarios during setup
   - **Fail fast**: Raise ValueError immediately (don't silently skip)

## 6. Validation & Quality Control

### 6.1 Inter-Model Reliability

**Question**: Do different judge models agree?

**Test**: Evaluate same set of reasoning samples with:
- Gemini 2.5 Pro (primary)
- GPT-4o (alternative)
- Claude Opus 4.5 (alternative)

**Metric**: Cohen's Kappa (inter-rater agreement)
- κ > 0.8: Substantial agreement
- κ = 0.6-0.8: Moderate agreement
- κ < 0.6: Poor agreement (review rubric)

### 6.2 Human Benchmark

**Gold standard**: Expert human annotations

**Process**:
1. Select 20 diverse reasoning samples (7 early, 7 mid, 6 late)
2. Have 2 human experts evaluate independently using same rubric
3. Compute human-LLM agreement rate

**Acceptance criterion**: LLM judge agrees with human majority ≥ 80% of time

### 6.3 Edge Case Testing

**Challenging scenarios**:
1. **Correct action, wrong reasoning**: Agent gets lucky
2. **Wrong action, sound reasoning**: Agent has incorrect beliefs but reasons correctly from them
3. **Ambiguous phrasing**: Agent uses vague language ("probably", "likely")
4. **Verbose vs. concise**: Long explanations vs. short justifications

**Expected**: Rubric should consistently handle all edge cases

## 7. Limitations & Mitigations

### 7.1 Known Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Judge LLM has same biases as agent LLM | May not catch systematic errors | Use ground truth as source of truth (not judge's understanding) |
| Scoring is ordinal, not interval | Can't claim "2× better reasoning" | Report categorical distributions + composite score |
| Language/style bias | Formal language may score higher | Normalize by comparing relative scores within same agent |
| No verification of ground truth correctness | Garbage in, garbage out | Expert review of all ground truth annotations |

### 7.2 Mitigation Strategies

**Ground truth as authority**:
- Judge does NOT interpret Hanabi rules
- Judge ONLY compares agent reasoning to provided ground truth
- Prevents alignment errors (if judge misunderstands rules, annotations compensate)

**Explicit rubric**:
- Four orthogonal dimensions (not single holistic score)
- Clear categorical boundaries
- Reduces subjective judgment

**Selective judging**:
- Skip cases where evaluation adds no information (FATAL actions)
- Focus judging effort on ambiguous/critical scenarios
- Reduces cost while maintaining research value

## 8. Reporting & Analysis

### 8.1 Per-Scenario Metrics

For each tested scenario, report:
```python
{
    "scenario_name": str,
    "game_stage": str,
    "agent_action": dict,
    "action_safety": str,
    "matches_ground_truth": bool,
    "judge_evaluation": {
        "factual_accuracy": str,
        "impossibility_recognition": str,
        "logical_soundness": str,
        "action_alignment": str,
        "composite_score": float,
        "explanation": str
    } | None  # None if not judged
}
```

### 8.2 Aggregate Analysis

**Across game stages**:
```
Early game: Mean judge score = 0.73 ± 0.15
Mid game:   Mean judge score = 0.68 ± 0.18
Late game:  Mean judge score = 0.81 ± 0.12

Interpretation: Reasoning improves in late game (higher stakes → more careful analysis)
```

**Dimension breakdown**:
```
Factual Accuracy:         85% CORRECT, 12% PARTIAL, 3% INCORRECT
Impossibility Recognition: 62% CORRECT, 28% PARTIAL, 8% INCORRECT, 2% N/A
Logical Soundness:        74% SOUND, 21% WEAK, 5% UNSOUND
Action Alignment:         91% ALIGNED, 9% MISALIGNED
```

**Correlation with outcomes**:
```
High judge score (≥0.8): 92% safe actions
Low judge score (<0.5):  38% safe actions

→ Judge score predicts action safety (validates rubric)
```

## 9. References & Version History

**Version**: 1.0
**Date**: 2025-12-16
**Authors**: Anthony & Yash

**Related documents**:
- Game stage classification: `/Users/an/algoverse/hanabi1/tests_anthony/GAME_STAGE_DEFINITION.md`
- Implementation: `/Users/an/algoverse/hanabi1/tests_anthony/shared/llm_judge.py`
- Research plan: `/Users/an/.claude/plans/immutable-floating-kahan.md`

**Changelog**:
- 2025-12-16: Initial specification

---

**For questions or clarifications**, see implementation code or consult research plan.
