# Logical Coherence Scoring - How It Works

## Overview

The logical coherence scoring system evaluates the **quality and structure** of the LLM agent's Chain of Thought (CoT) reasoning. It measures how well-organized, logically sound, and complete the agent's reasoning is when making decisions.

**Score Range:** 0.0 to 1.0 (higher is better)

## Why It Matters

Coherent reasoning indicates:
- ✅ The agent is actually "thinking" through the problem
- ✅ Decisions are based on logical analysis, not random guesses
- ✅ The reasoning can be audited and understood
- ✅ Poor reasoning patterns can be identified and fixed

## How The Score Is Calculated

Location: [tests_anthony/analysis.py](tests_anthony/analysis.py) - `check_logical_coherence()` function

### Scoring Components

The coherence score is built from **7 key indicators**:

#### 1. Causal Reasoning Indicators (+0.3 points)
**What:** Presence of causal/logical connectors
**Keywords:** "because", "since", "therefore", "thus", "so"
**Why:** Shows the agent is explaining WHY, not just WHAT

```python
if any(word in reasoning.lower() for word in ['because', 'since', 'therefore', 'thus', 'so']):
    score += 0.3
```

**Example:**
```
❌ Low coherence: "I will play card 1."
✅ High coherence: "I will play card 1 BECAUSE it matches the fireworks color."
```

#### 2. Analysis Structure (+0.2 points)
**What:** Presence of structured analysis sections
**Keywords:** "analysis", "consider", "evaluate", "assess"
**Why:** Shows systematic thinking, not ad-hoc reactions

```python
if any(word in reasoning.lower() for word in ['analysis', 'consider', 'evaluate', 'assess']):
    score += 0.2
```

**Example:**
```
✅ "Let me ANALYZE the current game state..."
✅ "I need to CONSIDER the following factors..."
```

#### 3. Evidence-Based Reasoning (+0.15 points)
**What:** Reference to concrete game state facts
**Keywords:** "fireworks", "stack", "hand", "card", "clue"
**Why:** Grounds reasoning in observable evidence

```python
if any(word in reasoning.lower() for word in ['fireworks', 'stack', 'hand', 'card', 'clue']):
    score += 0.15
```

**Example:**
```
✅ "The FIREWORKS show red is at 3, so red-4 is playable..."
✅ "My HAND contains 5 cards with no hints..."
```

#### 4. Safety Awareness (+0.15 points)
**What:** Consideration of risk and safety
**Keywords:** "safe", "risk", "danger", "careful", "avoid"
**Why:** Shows thoughtful decision-making, not reckless play

```python
if any(word in reasoning.lower() for word in ['safe', 'risk', 'danger', 'careful', 'avoid']):
    score += 0.15
```

**Example:**
```
✅ "Playing this card is RISKY because it might be critical..."
✅ "To be SAFE, I should discard instead..."
```

#### 5. Belief Graph Awareness (+0.1 points)
**What:** Reference to probabilistic beliefs or uncertainty
**Keywords:** "probability", "likely", "possible", "uncertain"
**Why:** Shows use of belief information (when provided)

```python
if any(word in reasoning.lower() for word in ['probability', 'likely', 'possible', 'uncertain']):
    score += 0.1
```

**Example:**
```
✅ "The PROBABILITY that this card is red-3 is 40%..."
✅ "It's LIKELY this card is playable given the hints..."
```

#### 6. Minimum Length Requirement (+0.1 points)
**What:** Reasoning has substantial content
**Threshold:** At least 20 words
**Why:** Prevents trivial/minimal explanations

```python
if len(reasoning.split()) >= 20:
    score += 0.1
```

**Example:**
```
❌ Low: "I discard." (2 words)
✅ High: "I will discard my leftmost card because it has no hints and we need to regain information tokens for strategic hints later in the game." (25 words)
```

#### 7. No Contradictions Penalty (-0.2 points)
**What:** Check for logical contradictions
**Keywords:** "but however", "although but", "despite though"
**Why:** Contradictory reasoning indicates confusion

```python
if any(phrase in reasoning.lower() for phrase in ['but however', 'although but', 'despite though']):
    score -= 0.2
```

**Example:**
```
❌ "This card is safe BUT HOWEVER it might be critical..."
```

## Score Interpretation

### Excellent (0.8 - 1.0)
- Structured, evidence-based reasoning
- Clear causal links
- Safety considerations
- Uses all available information
- **Action:** Trust this reasoning pattern

### Good (0.6 - 0.79)
- Solid reasoning with most elements present
- Minor gaps in structure or evidence
- **Action:** Generally reliable

### Moderate (0.4 - 0.59)
- Basic reasoning but missing key elements
- May lack causal explanations or evidence
- **Action:** Review decisions carefully

### Poor (0.2 - 0.39)
- Minimal reasoning
- Lacks structure or evidence
- **Action:** Investigate why reasoning is weak

### Critical (0.0 - 0.19)
- Nearly no reasoning or contradictory
- Decisions may be random
- **Action:** Agent needs prompt engineering fixes

## Real Example from Test

**Scenario:** Turn 1, Game Start
**Context Type:** Full Belief Graph

**Reasoning Text:**
```
**SITUATION ANALYSIS**
I have no hints on my cards. The fireworks are all at 0. We have full
information tokens available. Based on the belief graph probabilities, my
cards could be any color/rank combination.

**DECISION**
Since I have no information about my cards, I should discard to gain an
information token for strategic hints later. This is the SAFE choice because
we're early in the game and can afford to discard unknown cards.
```

**Coherence Score Breakdown:**
- ✅ Causal reasoning: "Since", "because" (+0.3)
- ✅ Analysis structure: "ANALYSIS" (+0.2)
- ✅ Evidence: "fireworks", "cards", "hints" (+0.15)
- ✅ Safety: "SAFE" (+0.15)
- ✅ Belief awareness: "probabilities" (+0.1)
- ✅ Length: 58 words (+0.1)
- ✅ No contradictions (+0.0)

**Total: 1.0 / 1.0 (Perfect)**

## How It's Used in Ablation Study

### 1. Per-Decision Scoring
Every agent decision gets a coherence score stored in metrics:

```python
coherence_score = check_logical_coherence(reasoning)

metrics = AgentDecisionMetrics(
    # ... other fields ...
    logical_coherence_score=coherence_score,
    reasoning_text=reasoning
)
```

### 2. Comparative Analysis
Compare coherence across context types:

```python
# Which context type produces best reasoning?
full_belief_graph:    avg coherence = 0.85
raw_game_state:       avg coherence = 0.72
minimal_context:      avg coherence = 0.45
```

**Insight:** More context → better reasoning structure

### 3. Correlation Analysis
Check if coherent reasoning → safe actions:

```python
safe_actions_high_coherence = 92%  # Coherence > 0.7
safe_actions_low_coherence = 34%   # Coherence < 0.4
```

**Insight:** Strong reasoning correlates with safe decisions

### 4. Debugging Agent Behavior
Identify patterns in poor reasoning:

```python
# Find low-coherence decisions
low_coherence = [m for m in metrics if m.logical_coherence_score < 0.4]

# Analyze what went wrong
for m in low_coherence:
    print(f"Scenario: {m.scenario_name}")
    print(f"Context: {m.context_type}")
    print(f"Reasoning: {m.reasoning_text}")
```

## Limitations

### What It Doesn't Measure

1. **Factual Correctness**
   - High coherence ≠ correct reasoning
   - Agent might be coherently wrong
   - **Solution:** Also check action safety metrics

2. **Depth of Analysis**
   - Counts keywords, not depth of thought
   - Simple reasoning can score high
   - **Solution:** Manual review of sample decisions

3. **Domain-Specific Logic**
   - Doesn't check Hanabi-specific strategies
   - Generic coherence only
   - **Solution:** Add custom Hanabi checks

4. **Cultural/Language Bias**
   - English-centric keywords
   - Formal language scores higher
   - **Solution:** Normalize for language style

## Improving Coherence Scores

### If Scores Are Low

**1. Prompt Engineering**
```python
# Add explicit structure requests
"Please provide your reasoning in the following format:
1. Situation Analysis (what do I observe?)
2. Options Evaluation (what could I do?)
3. Decision Justification (why this choice?)"
```

**2. Few-Shot Examples**
Show the agent examples of high-coherence reasoning in the prompt.

**3. Chain of Thought Prompting**
```python
"Let's think step by step..."
"First, analyze... Then, consider... Finally, decide..."
```

**4. Enforce Constraints**
```python
"Your reasoning must:
- Explain WHY, not just WHAT
- Reference specific game state facts
- Consider safety and risk
- Be at least 50 words"
```

## Related Metrics

Coherence is analyzed alongside:

1. **`reasoning_length_words`** - Quantity of reasoning
2. **`referenced_probabilities`** - Use of belief information
3. **`mentioned_impossible`** - Awareness of constraints
4. **`action_safety`** - Decision outcome quality
5. **`matches_ground_truth`** - Decision correctness

**Together these paint a complete picture of agent reasoning quality.**

## Code Reference

### Main Implementation
- **Function:** `check_logical_coherence(reasoning: str) -> float`
- **File:** [tests_anthony/analysis.py:295-327](tests_anthony/analysis.py#L295-L327)

### Usage in Tests
- **Function:** `analyze_reasoning_quality(reasoning: str) -> Dict`
- **File:** [tests_anthony/analysis.py:269-293](tests_anthony/analysis.py#L269-L293)

### Metrics Storage
- **Class:** `AgentDecisionMetrics`
- **Field:** `logical_coherence_score: float`
- **File:** [tests_anthony/metrics.py](tests_anthony/metrics.py)

### Report Generation
- **Function:** `generate_ablation_report(results: List[AgentDecisionMetrics]) -> str`
- **File:** [tests_anthony/reporting.py](tests_anthony/reporting.py)

## Future Enhancements

Potential improvements to the scoring system:

1. **Weighted Components**
   - Some indicators more important than others
   - Adaptive weights based on context type

2. **Negative Indicators**
   - Detect circular reasoning
   - Flag non-sequiturs
   - Penalize vague language ("probably", "maybe" without evidence)

3. **Domain-Specific Checks**
   - Hanabi strategy patterns
   - Convention adherence
   - Theory of mind reasoning

4. **ML-Based Scoring**
   - Train classifier on human-rated reasoning
   - More nuanced quality assessment
   - Detect sophisticated patterns

5. **Reasoning Graph Analysis**
   - Parse reasoning into logical graph
   - Check for gaps in reasoning chain
   - Measure argument strength

---

**Questions?** Check the code or examine sample outputs in `ablation_results.json` to see coherence scores in practice!
