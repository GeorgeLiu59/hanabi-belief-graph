"""LLM-as-Judge evaluation protocol for reasoning quality assessment.

Implements the formal scoring rubric from LLM_JUDGE_PROTOCOL.md.

Key components:
1. Selective judging (only evaluate when meaningful)
2. Four-dimensional rubric (FA, IR, LS, AA)
3. Ground truth comparison (prevents alignment errors)
4. Composite scoring with weighted dimensions

See LLM_JUDGE_PROTOCOL.md for complete specification and rationale.
"""

import json
import os
from typing import Dict, Any, Optional, Literal
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Type aliases for scoring dimensions
FactualAccuracy = Literal["CORRECT", "PARTIAL", "INCORRECT"]
ImpossibilityRecognition = Literal["CORRECT", "PARTIAL", "INCORRECT", "N/A"]
LogicalSoundness = Literal["SOUND", "WEAK", "UNSOUND"]
ActionAlignment = Literal["ALIGNED", "MISALIGNED"]


# ============================================================================
# Selective Judging Strategy
# ============================================================================

def should_judge(
    action_safety: str,
    matches_ground_truth: bool,
    scenario_name: str
) -> bool:
    """
    Determine if scenario warrants LLM-as-Judge evaluation.

    Cost-efficiency principle: Only judge when evaluation provides value.

    Args:
        action_safety: Safety classification ("SAFE", "RISKY", "FATAL")
        matches_ground_truth: Whether action matches expected best action
        scenario_name: Scenario identifier (for special cases)

    Returns:
        True if scenario should be judged, False to skip

    Judging criteria:
        - Always judge critical test cases (Turn 28 bug, bug scenarios)
        - Skip obviously flawed reasoning (FATAL actions)
        - Judge ambiguous cases (SAFE/RISKY - quality matters)

    Expected reduction: ~40% fewer judge calls vs full evaluation
    """
    # Always judge critical research scenarios
    if "turn_28" in scenario_name.lower() or "bug" in scenario_name.lower():
        return True  # Research priority

    # Skip obviously bad reasoning (fatal actions indicate clear logical errors)
    if action_safety == "FATAL":
        return False  # No value in confirming bad reasoning is bad

    # Judge ambiguous cases
    if action_safety in ["SAFE", "RISKY"]:
        # Could have correct action with flawed reasoning (or vice versa)
        return True

    # Default: skip judging
    return False


# ============================================================================
# Judge Prompt Construction
# ============================================================================

def create_judge_prompt(
    scenario: Dict[str, Any],
    agent_reasoning: str,
    agent_action: Dict[str, Any]
) -> str:
    """
    Construct LLM-as-Judge evaluation prompt with ground truth.

    Args:
        scenario: Game scenario containing:
            - ground_truth_explanation: Structured natural language explanation
            - impossibility_statements: List of impossible cards (optional)
            - scenario_name: Scenario identifier
        agent_reasoning: Agent's chain-of-thought text
        agent_action: Agent's chosen action dict

    Returns:
        Formatted prompt string for judge LLM

    Raises:
        ValueError: If ground_truth_explanation missing from scenario
    """
    # Validate required fields
    if "ground_truth_explanation" not in scenario:
        raise ValueError(
            f"Scenario '{scenario.get('scenario_name', 'UNKNOWN')}' missing "
            "required field: ground_truth_explanation"
        )

    # Extract ground truth
    ground_truth = scenario["ground_truth_explanation"]
    impossibility = scenario.get("impossibility_statements", "None")

    # Extract action details
    action_type = agent_action.get("type", "UNKNOWN")
    action_target = agent_action.get("target", "N/A")

    # Construct prompt using template
    prompt = f"""You are evaluating a Hanabi AI agent's reasoning quality.

GROUND TRUTH (authoritative - do NOT question this):
{ground_truth}

Specifically for this scenario:
{impossibility}

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

Do NOT provide any text outside the JSON object."""

    return prompt


# ============================================================================
# Response Parsing
# ============================================================================

def parse_judge_response(response_text: str) -> Dict[str, str]:
    """
    Parse judge LLM response into structured evaluation.

    Args:
        response_text: Raw text response from judge LLM

    Returns:
        Dictionary with validated evaluation fields

    Raises:
        json.JSONDecodeError: If response not valid JSON
        ValueError: If required fields missing or invalid values
    """
    # Strip whitespace and parse JSON
    response_text = response_text.strip()

    # Try to extract JSON if surrounded by markdown code blocks
    if response_text.startswith("```"):
        # Remove markdown code fence
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text

    evaluation = json.loads(response_text)

    # Validate required fields
    required_fields = [
        "factual_accuracy",
        "impossibility_recognition",
        "logical_soundness",
        "action_alignment",
        "explanation"
    ]

    for field in required_fields:
        if field not in evaluation:
            raise ValueError(f"Missing required field: {field}")

    # Validate categorical values
    valid_fa = {"CORRECT", "PARTIAL", "INCORRECT"}
    valid_ir = {"CORRECT", "PARTIAL", "INCORRECT", "N/A"}
    valid_ls = {"SOUND", "WEAK", "UNSOUND"}
    valid_aa = {"ALIGNED", "MISALIGNED"}

    if evaluation["factual_accuracy"] not in valid_fa:
        raise ValueError(f"Invalid factual_accuracy: {evaluation['factual_accuracy']}")
    if evaluation["impossibility_recognition"] not in valid_ir:
        raise ValueError(f"Invalid impossibility_recognition: {evaluation['impossibility_recognition']}")
    if evaluation["logical_soundness"] not in valid_ls:
        raise ValueError(f"Invalid logical_soundness: {evaluation['logical_soundness']}")
    if evaluation["action_alignment"] not in valid_aa:
        raise ValueError(f"Invalid action_alignment: {evaluation['action_alignment']}")

    return evaluation


# ============================================================================
# Composite Scoring
# ============================================================================

def calculate_judge_score(
    fa: FactualAccuracy,
    ir: ImpossibilityRecognition,
    ls: LogicalSoundness,
    aa: ActionAlignment,
    ground_truth: str
) -> float:
    """
    Compute weighted composite score for reasoning quality.

    Args:
        fa: Factual Accuracy ∈ {CORRECT, PARTIAL, INCORRECT}
        ir: Impossibility Recognition ∈ {CORRECT, PARTIAL, INCORRECT, N/A}
        ls: Logical Soundness ∈ {SOUND, WEAK, UNSOUND}
        aa: Action Alignment ∈ {ALIGNED, MISALIGNED}
        ground_truth: Ground truth explanation (used for N/A handling)

    Returns:
        Composite score ∈ [0.0, 1.0]

    Scoring formula:
        score = w_fa * v_fa + w_ir * v_ir + w_ls * v_ls + w_aa * v_aa

    Where:
        v_x ∈ {0.0, 0.5, 1.0} (value mapping)
        w_x = weight for dimension (sum to 1.0)

    Base weights:
        - FA: 0.3, IR: 0.4, LS: 0.2, AA: 0.1
        - If IR=N/A: FA: 0.5, IR: 0.0, LS: 0.2, AA: 0.1 (redistribute)
    """
    # Map categorical scores to numeric values
    fa_val = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}[fa]
    ls_val = {"SOUND": 1.0, "WEAK": 0.5, "UNSOUND": 0.0}[ls]
    aa_val = {"ALIGNED": 1.0, "MISALIGNED": 0.0}[aa]

    # Handle impossibility recognition (N/A case)
    if ir == "N/A":
        # No impossible cards in scenario - redistribute weight to FA
        # Heuristic: if ground truth mentions "impossible", scenario should have had them
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


# ============================================================================
# Main Evaluation Function
# ============================================================================

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
            "model_used": str,
            "success": bool
        }

    Raises:
        ValueError: If ground truth missing from scenario
        Exception: If API call fails after retries
    """
    # Create prompt
    prompt = create_judge_prompt(scenario, agent_reasoning, agent_action)

    # Initialize Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)

    # Call judge LLM with retry logic
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            response = llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,  # Low temperature for consistency
                    candidate_count=1
                )
            )

            response_text = response.text

            # Parse JSON response
            evaluation = parse_judge_response(response_text)

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
            evaluation["success"] = True

            return evaluation

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries:
                # Retry with clarified prompt
                prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON, no additional text."
                continue
            else:
                # All retries failed - return error result
                return {
                    "factual_accuracy": "INCORRECT",
                    "impossibility_recognition": "N/A",
                    "logical_soundness": "UNSOUND",
                    "action_alignment": "MISALIGNED",
                    "composite_score": 0.0,
                    "explanation": f"Judge evaluation failed: {str(e)}",
                    "model_used": model,
                    "success": False,
                    "error": str(e)
                }
        except Exception as e:
            if attempt < max_retries:
                continue
            else:
                raise  # Re-raise if all retries exhausted


# ============================================================================
# Utility Functions
# ============================================================================

def get_score_interpretation(score: float) -> str:
    """
    Get human-readable interpretation of composite score.

    Args:
        score: Composite score ∈ [0.0, 1.0]

    Returns:
        Interpretation string

    Score bands:
        0.9-1.0: Excellent reasoning
        0.7-0.89: Good reasoning
        0.5-0.69: Adequate reasoning
        0.3-0.49: Poor reasoning
        0.0-0.29: Critical reasoning failure
    """
    if score >= 0.9:
        return "Excellent reasoning - factually accurate, logically sound, recognizes constraints"
    elif score >= 0.7:
        return "Good reasoning - minor gaps but fundamentally sound"
    elif score >= 0.5:
        return "Adequate reasoning - significant issues but not critically flawed"
    elif score >= 0.3:
        return "Poor reasoning - major factual or logical errors"
    else:
        return "Critical reasoning failure - incorrect facts or unsound logic"


# TODO (for Yash/Anthony):
# 1. Add inter-model reliability testing (test same reasoning with multiple judge models)
# 2. Implement human benchmark comparison (compare LLM judge to expert annotations)
# 3. Add edge case handling tests (verbose vs concise reasoning, vague language, etc.)
# 4. Create validation suite for judge prompt effectiveness
# 5. Implement caching for repeated evaluations of same reasoning
