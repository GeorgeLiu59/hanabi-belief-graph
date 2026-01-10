"""
Single-turn CoT harness for hint scenarios.

Runs multiple scenario sets (different JSON files) and saves CoT outputs
to their corresponding output files.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from hanabi_learning_environment.agents.prompt_manager import PromptManager

# Mapping of scenario files to their corresponding output files
RUNS = [
    {
        "scenarios_path": Path(__file__).parent / "scenarios" / "jump_to_conclusions.json",
        "output_filename": "jump_to_conclusions.json",
    },
    # {
    #     "scenarios_path": Path(__file__).parent / "scenarios" / "scenarios_my_hand.json",
    #     "output_filename": "cot_outputs_my_hand.json",
    # },
    # {
    #     "scenarios_path": Path(__file__).parent / "scenarios" / "scenarios_full_beliefs.json",
    #     "output_filename": "cot_outputs_full_beliefs.json",
    # },
    # {
    #     "scenarios_path": Path(__file__).parent / "scenarios" / "scenarios_nl.json",
    #     "output_filename": "cot_outputs_nl.json",
    # },
]

MODEL_RUNS = [
    {"model_name": "gemini-2.5-flash", "output_dir": Path(__file__).parent / "cot_outputs_model_2-5"},
    #{"model_name": "gemini-2.0-flash", "output_dir": Path(__file__).parent / "cot_outputs_model_2"},
]


def load_scenarios(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def format_prompt(rules_text: str, scenario: dict) -> str:
    """Construct a self-contained prompt for a single-turn decision."""
    obs = scenario["observation"]
    history_lines = scenario.get("history_summary", [])
    history_text = "\n".join(f"- {line}" for line in history_lines) or "(none)"

    teammate_hand = ", ".join(obs.get("teammate_hand_visible", []))
    knowledge = "; ".join(obs.get("known_teammate_knowledge", [])) or "Unknown"

    prompt = f"""You are a cooperative Hanabi agent. Decide the next hint/action for a 2-player game.

## GAME RULES
{rules_text}

## SCENARIO ID
{scenario["id"]}

## GAME STATE (single turn)
- Current player: {obs.get("current_player")}
- Fireworks: {obs.get("fireworks")}
- Clues: {obs.get("clues")}
- Lives: {obs.get("lives")}
- Deck size: {obs.get("deck_size")}
- Teammate visible hand: {teammate_hand}
- Known teammate knowledge: {knowledge}

## HISTORY SUMMARY
{history_text}

## BELIEF GRAPH SUMMARY
{scenario.get("belief_graph_summary", "")}

## TASK
- Provide a concise chain-of-thought (reasoning) on what hint/action to take.
- Then output the action as JSON with keys: action_type, card_index, color, rank, target_player/target_offset (use null where not applicable).
- Focus on giving non-redundant, highest-value information based on what the teammate already knows.

Respond in this format:
REASONING:
<your reasoning>

ACTION:
{{"action_type": "...", "card_index": ..., "color": "...", "rank": ..., "target_player": "...", "target_offset": ...}}
"""
    return prompt


def save_outputs(outputs, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    merged = {**existing, **outputs}
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Saved/merged CoT outputs to {output_path}")


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Please export it or add to .env.")

    pm = PromptManager()
    rules = pm.get_hanabi_game_rules()

    for model_cfg in MODEL_RUNS:
        model_name = model_cfg["model_name"]
        output_dir = model_cfg["output_dir"]
        client = genai.Client(api_key=api_key)

        for run in RUNS:
            scenarios_path = run["scenarios_path"]
            output_path = output_dir / run["output_filename"]

            if not scenarios_path.exists():
                print(f"Skipping missing scenarios file: {scenarios_path}")
                continue

            scenarios = load_scenarios(scenarios_path)
            outputs = {}

            for scenario in scenarios:
                prompt = format_prompt(rules, scenario)
                print(f"[{model_name}][{scenarios_path.name}] Querying model for scenario: {scenario['id']}")
                response = client.models.generate_content(model=model_name, contents=prompt)
                outputs[scenario["id"]] = {
                    "prompt": prompt,
                    "response_text": response.text,
                    "model": model_name,
                }

            save_outputs(outputs, output_path)


if __name__ == "__main__":
    main()
