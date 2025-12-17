"""
Direct Gemini API adapter for natural language belief ablation study.

This adapter makes direct API calls to Gemini with natural language belief
descriptions, bypassing the complex agent infrastructure.
"""

from typing import Dict, Any, Tuple
import os
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()


def create_agent_decision_function(
    agent_type: str = "nl_belief",
    model: str = "gemini-2.0-flash-thinking-exp"
) -> callable:
    """
    Factory function to create Gemini-based decision function for ablation tests.

    Args:
        agent_type: Type of agent (currently only "nl_belief" supported)
        model: Gemini model to use

    Returns:
        Function matching signature: (scenario, nl_belief_context) -> (action, reasoning)
    """
    # Initialize Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment")

    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel(model)

    def decide(scenario: Dict[str, Any], nl_belief_context: str) -> Tuple[Dict[str, Any], str]:
        """
        Make decision using Gemini with natural language belief context.

        Args:
            scenario: Test scenario dict
            nl_belief_context: Natural language belief description

        Returns:
            (action_dict, reasoning_text)
        """
        # Build prompt with NL beliefs
        prompt = _build_decision_prompt(scenario, nl_belief_context)

        # Call Gemini
        try:
            response = llm.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Low for consistency
                    candidate_count=1
                )
            )

            response_text = response.text

            # Parse response
            action, reasoning = _parse_gemini_response(response_text, scenario)

            return action, reasoning

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            # Return safe fallback action
            return {
                "type": "DISCARD",
                "target": 0
            }, f"Error during API call: {str(e)}"

    return decide


def _build_decision_prompt(scenario: Dict[str, Any], nl_belief_context: str) -> str:
    """
    Build Gemini prompt with game state and NL belief context.

    Args:
        scenario: Game scenario dict
        nl_belief_context: Natural language belief description

    Returns:
        Formatted prompt string
    """
    # Extract game state
    fireworks = scenario['fireworks']
    life_tokens = scenario['life_tokens']
    info_tokens = scenario.get('info_tokens', scenario.get('information_tokens', 8))
    deck_size = scenario.get('deck_size', 50)
    turn = scenario.get('turn', 1)

    # Format partner's visible hand
    partner_hand = scenario['player_hands']['player_1']
    partner_hand_str = ", ".join([
        f"{card['color']}-{card['rank']}" for card in partner_hand
    ])

    # Get agent's hand size
    agent_hand_size = len(scenario['player_hands']['player_0'])

    prompt = f"""You are playing Hanabi, a cooperative card game. You are Player 0.

# Current Game State

**Turn {turn}**

**Fireworks (cards successfully played):**
- Red: {fireworks['R']}
- Yellow: {fireworks['Y']}
- Green: {fireworks['G']}
- White: {fireworks['W']}
- Blue: {fireworks['B']}

**Resources:**
- Life tokens: {life_tokens}/3
- Information tokens: {info_tokens}/8
- Deck remaining: {deck_size} cards

**Partner's hand (Player 1 - visible to you):**
{partner_hand_str}

**Your hand (Player 0 - you cannot see your own cards):**
You hold {agent_hand_size} cards in positions 0-{agent_hand_size-1}.

---

# Your Beliefs About Your Hand

{nl_belief_context}

---

# Your Task

Choose ONE action from the following options:

1. **PLAY <position>** - Play a card from your hand (positions 0-{agent_hand_size-1})
2. **DISCARD <position>** - Discard a card from your hand (positions 0-{agent_hand_size-1})
3. **REVEAL_COLOR <color> to Player 1** - Tell partner about color (red/yellow/green/white/blue)
4. **REVEAL_RANK <rank> to Player 1** - Tell partner about rank (1-5)

# Response Format

Respond with EXACTLY this structure:

```
REASONING:
[Your chain-of-thought reasoning about the game state and which action to take]

ACTION:
[Your chosen action in one of these formats:]
PLAY <position>
DISCARD <position>
REVEAL_COLOR <color>
REVEAL_RANK <rank>
```

**CRITICAL RULES:**
- Only play cards you are confident are playable (next card for a stack)
- If life tokens are low (1-2), be extra cautious about playing
- Use the impossibility constraints from your beliefs to make safer decisions
- Your reasoning should reference the fireworks state and your beliefs

Begin your response now:"""

    return prompt


def _parse_gemini_response(response_text: str, scenario: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Parse Gemini response into action dict and reasoning text.

    Args:
        response_text: Raw Gemini response
        scenario: Game scenario (for fallback)

    Returns:
        (action_dict, reasoning_text)
    """
    # Extract reasoning
    reasoning_match = re.search(r'REASONING:\s*(.+?)(?=ACTION:|$)', response_text, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    else:
        reasoning = response_text  # Use full response as reasoning

    # Extract action
    action_match = re.search(r'ACTION:\s*(.+?)(?=$|```)', response_text, re.DOTALL | re.IGNORECASE)
    if action_match:
        action_str = action_match.group(1).strip()
    else:
        # Try to find action keywords in the response
        action_str = response_text

    # Parse action string
    action = _parse_action_string(action_str, scenario)

    return action, reasoning


def _parse_action_string(action_str: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse action string into structured action dict.

    Args:
        action_str: Action string like "PLAY 0" or "REVEAL_COLOR red"
        scenario: Game scenario (for fallback)

    Returns:
        Action dict with 'type' and relevant fields
    """
    action_str = action_str.strip().upper()

    # Try PLAY
    play_match = re.search(r'PLAY\s+(\d+)', action_str)
    if play_match:
        return {
            "type": "PLAY",
            "target": int(play_match.group(1))
        }

    # Try DISCARD
    discard_match = re.search(r'DISCARD\s+(\d+)', action_str)
    if discard_match:
        return {
            "type": "DISCARD",
            "target": int(discard_match.group(1))
        }

    # Try REVEAL_COLOR
    color_match = re.search(r'REVEAL_COLOR\s+(\w+)', action_str)
    if color_match:
        color = color_match.group(1).lower()
        # Map to single letter
        color_map = {'red': 'R', 'yellow': 'Y', 'green': 'G', 'white': 'W', 'blue': 'B'}
        return {
            "type": "REVEAL_COLOR",
            "target_offset": 1,  # Reveal to partner (player 1)
            "color": color_map.get(color, color.upper()[0])
        }

    # Try REVEAL_RANK
    rank_match = re.search(r'REVEAL_RANK\s+(\d+)', action_str)
    if rank_match:
        return {
            "type": "REVEAL_RANK",
            "target_offset": 1,  # Reveal to partner (player 1)
            "rank": int(rank_match.group(1))
        }

    # Fallback: safe discard
    print(f"Warning: Could not parse action from: {action_str}")
    print("Falling back to DISCARD 0")
    return {
        "type": "DISCARD",
        "target": 0
    }
