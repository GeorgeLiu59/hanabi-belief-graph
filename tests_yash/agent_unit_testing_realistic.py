"""
Realistic single-turn CoT harness that matches actual game prompt structure.

This version mimics the exact prompt format used in the actual game, including:
- Belief graph natural language formatting
- Available actions enumeration
- YOUR HAND section
- Structured CoT requirements
- A/B testing between old (biased) and new (risk-aware) prompts
- WITH vs WITHOUT belief graph comparison

USAGE:
    # Run all combinations (default)
    python agent_unit_testing_realistic.py

    # Test only WITH belief graph, new CoT
    python agent_unit_testing_realistic.py --belief-mode with --cot-mode new

    # Test only WITHOUT belief graph, new CoT
    python agent_unit_testing_realistic.py --belief-mode without --cot-mode new

    # Compare WITH vs WITHOUT (both CoT modes)
    python agent_unit_testing_realistic.py --belief-mode both --cot-mode both

    # Specific scenario file
    python agent_unit_testing_realistic.py --scenarios scenarios/early_hint_scenarios.json --belief-mode both
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

from hanabi_learning_environment.agents.prompt_manager import PromptManager

# Mapping of scenario files to their corresponding output files
RUNS = [
    # {
    #     "scenarios_path": Path(__file__).parent / "scenarios" / "jump_to_conclusions.json",
    #     "output_filename": "jump_realistic2.json",
    # },
    # {
    #     "scenarios_path": Path(__file__).parent / "scenarios" / "scenarios_my_hand.json",
    #     "output_filename": "my_hand_realistic.json",
    # },
    {
        "scenarios_path": Path(__file__).parent / "scenarios" / "scenarios_full_beliefs.json",
        "output_filename": "full_beliefs_realistic.json",
    },
    
]

MODEL_RUNS = [
    {"model_name": "gemini-2.5-flash", "output_dir": Path(__file__).parent / "cot_outputs_model_2-5"},
]


def load_scenarios(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def format_beliefs_natural_language(beliefs: dict, is_teammate: bool = False, teammate_visible: list = None) -> str:
    """Format beliefs in natural language like actual game."""
    if not beliefs:
        return "(No belief data)"

    lines = []
    for card_id, belief in sorted(beliefs.items()):
        # Extract card number from ID (e.g., "P2_Card1" -> "1")
        card_num = card_id.split('_')[-1].replace('Card', '') if '_Card' in card_id else card_id

        # Get color and rank distributions
        if 'agent_belief' in belief:
            # Nested structure (from test scenarios)
            color_dist = belief['agent_belief'].get('color_distribution', {})
            rank_dist = belief['agent_belief'].get('rank_distribution', {})
        else:
            # Flat structure
            color_dist = belief.get('color_distribution', {})
            rank_dist = belief.get('rank_distribution', {})

        # Handle string format like "uniform (20% each)"
        if isinstance(color_dist, str):
            color_dist = {c: 0.20 for c in ['red', 'blue', 'green', 'white', 'yellow']}
        if isinstance(rank_dist, str):
            rank_dist = {str(i): 0.20 for i in range(1, 6)}

        # Format colors with percentages
        color_items = [(c, p) for c, p in color_dist.items() if p > 0]
        color_items.sort(key=lambda x: -x[1])  # Sort by probability descending
        color_text = ", ".join([f"{c.upper()}({int(p*100)}%)" for c, p in color_items[:5]])

        # Format ranks with percentages
        rank_items = [(r, p) for r, p in rank_dist.items() if p > 0]
        rank_items.sort(key=lambda x: -x[1])
        rank_text = ", ".join([f"{r}({int(p*100)}%)" for r, p in rank_items[:5]])

        # Get max confidence
        max_color_conf = max(color_dist.values()) if color_dist else 0
        max_rank_conf = max(rank_dist.values()) if rank_dist else 0

        # Add header with actual card if teammate
        if is_teammate and teammate_visible:
            try:
                idx = int(card_num) - 1
                if 0 <= idx < len(teammate_visible):
                    actual_card = teammate_visible[idx]
                    lines.append(f"  - Card {card_num}: I see {actual_card}")
                else:
                    lines.append(f"  - Card {card_num}:")
            except (ValueError, IndexError):
                lines.append(f"  - Card {card_num}:")
        else:
            lines.append(f"  - Card {card_num} (card_index {int(card_num)-1}):")

        lines.append(f"    * Colors: {color_text}")
        lines.append(f"    * Ranks: {rank_text}")

        # Status line
        if max_color_conf == 1.0 and max_rank_conf == 1.0:
            top_color = max(color_dist.items(), key=lambda x: x[1])[0]
            top_rank = max(rank_dist.items(), key=lambda x: x[1])[0]
            lines.append(f"    * Status: Certain ({top_color.upper()} {top_rank})")
        elif max_rank_conf == 1.0:
            top_rank = max(rank_dist.items(), key=lambda x: x[1])[0]
            lines.append(f"    * Status: 100% rank known ({top_rank}), color uncertain")
        elif max_color_conf == 1.0:
            top_color = max(color_dist.items(), key=lambda x: x[1])[0]
            lines.append(f"    * Status: 100% color known ({top_color.upper()}), rank uncertain")
        else:
            lines.append(f"    * Status: {int(max_color_conf*100)}% color, {int(max_rank_conf*100)}% rank - Uncertain")

    return "\n".join(lines)


def build_available_actions(obs: dict) -> str:
    """Build available actions list like actual game."""
    actions = []
    idx = 0

    # Discards (0-4)
    for i in range(5):
        actions.append(f"{idx}: Discard card {i} (+1 info token)")
        idx += 1

    # Plays (5-9)
    for i in range(5):
        actions.append(f"{idx}: Play card {i}")
        idx += 1

    # Hints - colors (only for cards present in teammate hand)
    teammate_hand = obs.get('teammate_hand_visible', [])
    colors_in_hand = set()
    for card in teammate_hand:
        if isinstance(card, str) and len(card) >= 1:
            colors_in_hand.add(card[0])

    for color in sorted(colors_in_hand):
        current_player = obs.get('current_player', 'P1')
        target_player = 1 if current_player == 'P2' else 2
        actions.append(f"{idx}: Hint color {color} to player {target_player} (-1 info token)")
        idx += 1

    # Hints - ranks (only for ranks present)
    ranks_in_hand = set()
    for card in teammate_hand:
        if isinstance(card, str) and len(card) >= 2:
            ranks_in_hand.add(card[1])

    for rank in sorted(ranks_in_hand):
        current_player = obs.get('current_player', 'P1')
        target_player = 1 if current_player == 'P2' else 2
        actions.append(f"{idx}: Hint rank {rank} to player {target_player} (-1 info token)")
        idx += 1

    return "\n".join(actions)


def build_my_hand_section(my_beliefs: dict) -> str:
    """Build YOUR HAND section like actual game."""
    if not my_beliefs:
        return "(No hand data)"

    lines = []
    for card_id in sorted(my_beliefs.keys()):
        # Extract card number
        card_num = card_id.split('_')[-1].replace('Card', '') if '_Card' in card_id else card_id
        belief = my_beliefs[card_id]

        # Get distributions
        if 'agent_belief' in belief:
            color_dist = belief['agent_belief'].get('color_distribution', {})
            rank_dist = belief['agent_belief'].get('rank_distribution', {})
        else:
            color_dist = belief.get('color_distribution', {})
            rank_dist = belief.get('rank_distribution', {})

        # Handle string format
        if isinstance(color_dist, str):
            color_dist = {}
        if isinstance(rank_dist, str):
            rank_dist = {}

        # Determine what player knows
        max_color_conf = max(color_dist.values()) if color_dist else 0
        max_rank_conf = max(rank_dist.values()) if rank_dist else 0

        if max_color_conf == 1.0:
            color_known = max(color_dist.items(), key=lambda x: x[1])[0].upper()
        else:
            color_known = "unknown"

        if max_rank_conf == 1.0:
            rank_known = max(rank_dist.items(), key=lambda x: x[1])[0]
        else:
            rank_known = "unknown"

        lines.append(f"Card {card_num}: Color: {color_known}, Rank: {rank_known}")

    return "\n".join(lines)


def format_fireworks(fireworks: dict) -> str:
    """Format fireworks progress."""
    return "\n".join([f"- {color}: {level}/5" for color, level in sorted(fireworks.items())])


def format_discard_pile(discard: list) -> str:
    """Format discard pile."""
    if not discard:
        return "empty"

    formatted = []
    for item in discard:
        if isinstance(item, dict):
            formatted.append(f"{item.get('color', '?')}{item.get('rank', '?')}")
        else:
            formatted.append(str(item))

    return "\n".join([f"- {card}" for card in formatted]) if formatted else "empty"


def get_risk_aware_cot_prompt(lives: int) -> str:
    """Get the NEW risk-aware CoT prompt."""
    if lives == 1:
        threshold = "98%+ (CRITICAL: Last life!)"
        guidance = "With 1 life remaining, prefer enabling teammate plays or discarding over risky self-plays."
    elif lives == 2:
        threshold = "90%+ (Caution advised)"
        guidance = "With 2 lives, balance progress with safety."
    else:
        threshold = "80%+ (Normal risk tolerance)"
        guidance = "With 3 lives, calculated risks are acceptable."

    return f"""

## DECISION REQUIRED:
Choose your action.

## CHAIN OF THOUGHT REQUIREMENT:

Before providing your final JSON action, you MUST provide detailed reasoning following this structure:

**REASONING:**
**PART 1: BELIEF GRAPH ANALYSIS**
Your belief graph provides probabilistic information about card identities. Analyze it critically:

1. **Belief Graph Data**: What does your belief graph indicate?
   - Card probabilities/certainties?
   - Teammate knowledge patterns?
   - **IMPORTANT**: Belief graphs can have errors. Cross-check with observations.

2. **Risk Assessment** (CRITICAL with {lives} lives remaining):
   - Required confidence threshold: {threshold}
   - If playing a card: What's the probability of failure? Is this acceptable given lives remaining?
   - Safer alternatives: Can teammate make a certain play? Can you give a hint instead?
   - {guidance}

3. **Decision Process**:
   - What action best balances progress with safety?
   - How does belief confidence compare to required threshold?
   - If confidence is insufficient, what's the safest alternative?

**PART 2: QUICK DECISION SUMMARY**
Brief summary of action, confidence level, and risk justification.

**FORMAT:**
First provide your belief graph analysis in PART 1.
Then provide a brief summary in PART 2.
Then provide your action as a JSON object.

Example (with 1 life remaining):
REASONING:
**PART 1: BELIEF GRAPH ANALYSIS**
1. **Belief Graph Data**: My card at index 0 shows 85% RED, rank 3 (100%). Red firework is at R2.

2. **Risk Assessment**:
   - Required threshold: 98%+ (last life)
   - Current confidence: 85% RED = 15% chance of failure
   - 15% chance of game over is UNACCEPTABLE with 1 life
   - Safer alternative: Teammate has B1 (visible), which is 100% certain play

3. **Decision Process**: 85% < 98% required. Enable teammate's certain B1 play instead of risking game over.

**PART 2: QUICK DECISION SUMMARY**
85% confidence insufficient for last life. Hint rank 1 to enable teammate's certain B1 play.

{{"action_type": "REVEAL_RANK", "rank": 1, "target_offset": 1}}
"""


def get_original_biased_cot_prompt() -> str:
    """Get the ORIGINAL biased CoT prompt for comparison."""
    return """

## DECISION REQUIRED:
Choose your action.

## CHAIN OF THOUGHT REQUIREMENT:

Before providing your final JSON action, you MUST provide detailed reasoning following this structure:

**REASONING:**
**PART 1: BELIEF GRAPH ANALYSIS - YOUR DECISION ENGINE**
CRITICAL: Your belief graph is your primary decision-making tool. Analyze it thoroughly:

1. **Belief Graph Data**: What specific information from your belief graph drives this decision?
   - Card probabilities/certainties?
   - Teammate knowledge patterns?
   - Team_Focus or ToM insights?
   - How does this compare to raw observation?

2. **Decision Process**: How did belief graph information lead to your action?
   - What did the belief graph reveal that raw observation missed?
   - How did it change your strategic priorities?
   - Which data points were decisive and why?

**PART 2: QUICK DECISION SUMMARY**
Brief summary of action and key belief graph influence.

**FORMAT:**
First provide your belief graph analysis in PART 1.
Then provide a brief summary in PART 2.
Then provide your action as a JSON object.

Example:
REASONING:
**PART 1: BELIEF GRAPH ANALYSIS - YOUR DECISION ENGINE**
1. **Belief Graph Data**: My belief graph shows Team_Focus is PLAY_ONES (70% confidence) and P2 has 85% confidence in Red cards. This reveals the team prioritizes foundation building, making color clues most valuable now. My hand shows uniform probabilities (20% each), confirming I must share information.

2. **Decision Process**: The belief graph's Team_Focus information directly prioritized color clues over other options. The confidence levels confirmed Red information would be most valuable to support the PLAY_ONES strategy. This led me to choose a Red color hint instead of other alternatives.

**PART 2: QUICK DECISION SUMMARY**
Team_Focus (PLAY_ONES) + Red confidence (85%) → Red color hint to P2 to enable play.

{{"action_type": "REVEAL_COLOR", "color": "R", "target_offset": 1}}
"""


def format_prompt(rules_text: str, scenario: dict, use_new_cot: bool = True, include_belief_graph: bool = True) -> str:
    """Construct a prompt matching actual game structure.

    Args:
        rules_text: Game rules text
        scenario: Scenario dictionary
        use_new_cot: If True, use new risk-aware CoT prompt; else use old biased prompt
        include_belief_graph: If True, include belief graph sections; else omit for baseline comparison
    """
    obs = scenario["observation"]
    history_lines = scenario.get("history_summary", [])
    history_text = "\n".join(f"- {line}" for line in history_lines) or "(none)"

    belief_summary = scenario.get("belief_graph_summary", {})

    # Format belief graph in natural language (like actual game)
    my_hand_beliefs_nl = format_beliefs_natural_language(
        belief_summary.get("My_Hand_Beliefs", {})
    )
    teammate_beliefs_nl = format_beliefs_natural_language(
        belief_summary.get("Teammate_Hand_Beliefs", {}),
        is_teammate=True,
        teammate_visible=obs.get("teammate_hand_visible", [])
    )

    # Build available actions list (like actual game)
    available_actions = build_available_actions(obs)

    # Build YOUR HAND section - different for with/without belief graph
    if include_belief_graph:
        my_hand_section = build_my_hand_section(belief_summary.get("My_Hand_Beliefs", {}))
    else:
        # Without belief graph, agent knows nothing about their own hand
        my_hand_section = "\n".join([f"Card {i+1}: Color: unknown, Rank: unknown" for i in range(5)])

    current_player = obs.get("current_player", "P1")
    lives = obs.get("lives", 3)

    # Select CoT prompt based on flag
    if use_new_cot:
        cot_section = get_risk_aware_cot_prompt(lives)
    else:
        cot_section = get_original_biased_cot_prompt()

    # Determine other player number
    other_player = 1 if current_player == "P2" else 2

    # Build belief graph section (only if included)
    if include_belief_graph:
        belief_section = f"""
## BELIEF GRAPH ANALYSIS (PROBABILISTIC VARIANT)

**Game State:** {obs.get('clues', 0)} clues, {obs.get('lives', 3)} lives, {obs.get('deck_size', 50)} cards in deck

**MY HAND BELIEFS:**
{my_hand_beliefs_nl}

**TEAMMATE KNOWLEDGE MODEL:**

Player {other_player}'s cards:
{teammate_beliefs_nl}



**BELIEF GRAPH INTELLIGENCE (PROBABILISTIC VARIANT):**
Your belief graph contains quantified confidence distributions from aggregated historical evidence.

**Available Information:**
- Confidence levels for each color/rank possibility
- Information-theoretic value of potential actions
- Expected risk/reward calculations for different choices
- Uncertainty reduction potentials

**Strategic Applications:**
- High-confidence cards (>95%) can be played for critical points
- Medium-confidence cards (70-95%) work well for information gathering
- Clues can target cards with highest uncertainty reduction value
- Risk calculations inform optimal play/discard timing decisions

"""
    else:
        belief_section = f"""
## GAME STATE OBSERVATION (NO BELIEF TRACKING)

**Game State:** {obs.get('clues', 0)} clues, {obs.get('lives', 3)} lives, {obs.get('deck_size', 50)} cards in deck

**IMPORTANT:** You do NOT have access to a belief graph system. You can only see:
- The current game state (fireworks, clues, lives, deck size)
- Your teammate's visible cards
- The history of what has been played/discarded
- What hints have been given this turn (if shown in history)

You CANNOT see your own cards, and you have NO MEMORY of prior hints given to you unless explicitly stated in the history.

"""

    prompt = f"""You are a cooperative Hanabi agent playing Hanabi-Full-CardKnowledge.

## GAME RULES
{rules_text}

## SCENARIO ID
{scenario["id"]}

## HISTORY SUMMARY
{history_text}

{belief_section}

## FORMAT REQUIREMENTS:
Respond with ONLY a valid JSON object in this exact format (ranks MUST use human numbers 1-5):
{{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 1-5, "target_offset": 1-3}}

**IMPORTANT**:
- For PLAY/DISCARD: use "card_index" (0-4), set color/rank/target_offset to null
- For REVEAL_COLOR: use "color" and "target_offset", set card_index/rank to null
- For REVEAL_RANK: use "rank" (1-5) and "target_offset", set card_index/color to null

## CURRENT GAME STATE:

## GAME OVERVIEW
- Information tokens: {obs.get('clues', 0)}/8
- Life tokens: {obs.get('lives', 3)}/3
- Deck size: {obs.get('deck_size', 50)}
- Number of players: 2

## FIREWORKS PROGRESS
{format_fireworks(obs.get('fireworks', {}))}

## DISCARD PILE
{format_discard_pile(obs.get('discard_pile', []))}

## OTHER PLAYERS' HANDS
Player {other_player}: {' '.join(obs.get('teammate_hand_visible', []))}

## YOUR HAND
{my_hand_section}

## AVAILABLE ACTIONS
{available_actions}

{cot_section}
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Hanabi agent WITH and WITHOUT belief graph scaffolding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all combinations (default)
  python agent_unit_testing_realistic.py

  # Test only WITH belief graph, new CoT prompt
  python agent_unit_testing_realistic.py --belief-mode with --cot-mode new

  # Test only WITHOUT belief graph, new CoT prompt
  python agent_unit_testing_realistic.py --belief-mode without --cot-mode new

  # Compare WITH vs WITHOUT using new CoT prompt only
  python agent_unit_testing_realistic.py --belief-mode both --cot-mode new

  # Test specific scenario file
  python agent_unit_testing_realistic.py --scenarios scenarios/early_hint_scenarios.json
        """
    )

    parser.add_argument(
        "--belief-mode",
        choices=["with", "without", "both"],
        default="both",
        help="Whether to include belief graph: 'with' (include), 'without' (exclude), or 'both' (test both)"
    )

    parser.add_argument(
        "--cot-mode",
        choices=["new", "old", "both"],
        default="both",
        help="CoT prompt to use: 'new' (risk-aware), 'old' (biased), or 'both' (test both)"
    )

    parser.add_argument(
        "--scenarios",
        type=Path,
        help="Path to specific scenario file (if not provided, uses RUNS list)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. Please export it or add to .env.")

    pm = PromptManager()
    rules = pm.get_hanabi_game_rules()

    # Configure CoT modes based on args
    if args.cot_mode == "new":
        cot_modes = [{"name": "new_risk_aware", "use_new_cot": True}]
    elif args.cot_mode == "old":
        cot_modes = [{"name": "old_biased", "use_new_cot": False}]
    else:  # both
        cot_modes = [
            {"name": "new_risk_aware", "use_new_cot": True},
            {"name": "old_biased", "use_new_cot": False}
        ]

    # Configure belief modes based on args
    if args.belief_mode == "with":
        belief_modes = [{"name": "with_belief", "include_belief": True}]
    elif args.belief_mode == "without":
        belief_modes = [{"name": "without_belief", "include_belief": False}]
    else:  # both
        belief_modes = [
            {"name": "with_belief", "include_belief": True},
            {"name": "without_belief", "include_belief": False}
        ]

    # Configure scenario runs
    if args.scenarios:
        runs = [{
            "scenarios_path": args.scenarios,
            "output_filename": args.scenarios.stem + "_output.json"
        }]
    else:
        runs = RUNS

    # Print configuration
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"Belief modes: {[m['name'] for m in belief_modes]}")
    print(f"CoT modes: {[m['name'] for m in cot_modes]}")
    print(f"Scenarios: {[str(r['scenarios_path']) for r in runs]}")
    print("="*80 + "\n")

    for model_cfg in MODEL_RUNS:
        model_name = model_cfg["model_name"]
        output_dir = model_cfg["output_dir"]
        client = genai.Client(api_key=api_key)

        for belief_mode in belief_modes:
            for cot_mode in cot_modes:
                print(f"\n{'='*80}")
                print(f"Testing: {belief_mode['name']} + {cot_mode['name']}")
                print(f"{'='*80}\n")

                for run in runs:
                    scenarios_path = run["scenarios_path"]

                    # Modify output filename to include both belief mode and CoT mode
                    base_filename = run["output_filename"]
                    output_filename = base_filename.replace(
                        ".json",
                        f"_{belief_mode['name']}_{cot_mode['name']}.json"
                    )
                    output_path = output_dir / output_filename

                    if not scenarios_path.exists():
                        print(f"⚠️  Skipping missing scenarios file: {scenarios_path}")
                        continue

                    scenarios = load_scenarios(scenarios_path)
                    outputs = {}

                    for scenario in scenarios:
                        prompt = format_prompt(
                            rules,
                            scenario,
                            use_new_cot=cot_mode['use_new_cot'],
                            include_belief_graph=belief_mode['include_belief']
                        )
                        print(f"🔄 [{model_name}][{belief_mode['name']}][{cot_mode['name']}][{scenarios_path.name}] "
                              f"Querying for: {scenario['id']}")

                        try:
                            response = client.models.generate_content(model=model_name, contents=prompt)
                            outputs[scenario["id"]] = {
                                "prompt": prompt,
                                "response_text": response.text,
                                "model": model_name,
                                "cot_mode": cot_mode['name'],
                                "belief_mode": belief_mode['name']
                            }
                            print(f"✅ Completed: {scenario['id']}")
                        except Exception as e:
                            print(f"❌ Error on {scenario['id']}: {e}")
                            outputs[scenario["id"]] = {
                                "prompt": prompt,
                                "response_text": f"ERROR: {str(e)}",
                                "model": model_name,
                                "cot_mode": cot_mode['name'],
                                "belief_mode": belief_mode['name'],
                                "error": str(e)
                            }

                    save_outputs(outputs, output_path)
                    print(f"💾 Saved results to {output_path}\n")

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print("\nResults saved:")
    for model_cfg in MODEL_RUNS:
        output_dir = model_cfg["output_dir"]
        for run in runs:
            base_filename = run["output_filename"]
            for belief_mode in belief_modes:
                for cot_mode in cot_modes:
                    output_filename = base_filename.replace(
                        ".json",
                        f"_{belief_mode['name']}_{cot_mode['name']}.json"
                    )
                    print(f"  - {output_dir / output_filename}")


if __name__ == "__main__":
    main()
