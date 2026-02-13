"""Chain of Thought reasoning module for Hanabi agents."""

from typing import Dict, Any, List


class ChainOfThoughtReasoner:
    """Handles Chain of Thought reasoning for agent decisions."""
    
    def create_cot_prompt(self, observation: Dict[str, Any], base_prompt: str) -> str:
        """Create a prompt that asks for Chain of Thought reasoning."""
        cot_prompt = base_prompt + """

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

{"action_type": "REVEAL_COLOR", "color": "R", "target_offset": 1}
"""
        return cot_prompt

    def create_belief_update_cot_prompt(self, observation: Dict[str, Any], base_prompt: str) -> str:
        """Create a Chain of Thought prompt specifically for belief updates."""
        cot_prompt = base_prompt + """

## CHAIN OF THOUGHT BELIEF UPDATE REQUIREMENT:

CRITICAL: Before providing your final JSON belief graph update, you MUST provide detailed Chain of Thought reasoning showing how EVERY analysis component directly drives belief graph updates.

**REASONING:**

**PART 1: DIRECT IMPACT ANALYSIS** (Most Critical)
- What specific information did the clue provide?
- Which cards received direct updates (100% certain or negative inference)?
- **BELIEF GRAPH IMPACT**: How this updates My_Hand_Beliefs/Teammate_Hand_Beliefs with new color/rank certainties

**PART 2: GAME STAGE & TEAM STRATEGY REASONING**
- What game stage is this? (early/mid/late based on remaining cards, clues, lives)
- What is the team's current priority? (building fireworks, saving important cards, managing information)
- What is {giver} trying to communicate? (which card to play/save, signaling strategy)
- How does this help {recipient} make better decisions?
- **BELIEF GRAPH IMPACT**: How this updates Team_Focus distribution and strategic priorities in the belief graph

**PART 3: SHARED MENTAL MODEL REASONING**
- How should this update affect the team's shared mental model?
- What does this reveal about {giver}'s assessment of game state?
- How does this change what {recipient} knows about their cards?
- **BELIEF GRAPH IMPACT**: How this updates mental models of teammates in Teammate_Hand_Beliefs and ToM_Layer

**PART 4: THEORY OF MIND REASONING** (Theory of Mind variants only)
- First-order: What does this reveal about recipient's knowledge state?
- Recursive: What does giver think about recipient's knowledge of ME?
- Strategic: How does this affect team coordination and shared understanding?
- **BELIEF GRAPH IMPACT**: How this updates Recursive_ToM beliefs about what others think about MY skill/strategy

**PART 5: COORDINATION IMPLICATIONS**
- What future actions are now enabled/constrained?
- How does this affect risk assessment for upcoming decisions?
- What coordination opportunities does this create?
- **BELIEF GRAPH IMPACT**: How this updates inferred_skill, play_aggressiveness, hint_quality in ToM_Layer

**CRITICAL REQUIREMENT**: For each part above, you MUST explicitly state how the reasoning drives specific belief graph updates. Every analysis point must connect to concrete JSON structure changes.

**FORMAT:**
First provide your Chain of Thought analysis in the REASONING section above with explicit BELIEF GRAPH IMPACT connections.
Then provide your updated belief graph as a JSON object.

Example:
REASONING:

**PART 1: DIRECT IMPACT ANALYSIS** (Most Critical)
- The Red color clue to P2 cards 2,4 confirms P2_Card2 and P2_Card4 are Red (100% certain)
- P2's other cards (1,3,5) are definitively NOT Red (negative inference)
- **BELIEF GRAPH IMPACT**: Update P2_Card2 and P2_Card4 in Teammate_Hand_Beliefs with color="R", rank_probabilities, confidence=1.0. Set Red probability to 0.0 for P2_Card1, P2_Card3, P2_Card5.

**PART 2: GAME STAGE & TEAM STRATEGY REASONING**
- Mid-game with 15 cards remaining, 6 clues, 3 lives suggests focus on building fireworks
- Red clue indicates team needs Red fireworks built
- **BELIEF GRAPH IMPACT**: Increase Team_Focus "building_fireworks" probability from 0.3 to 0.6, decrease "saving_cards" to 0.2.

**PART 3: THEORY OF MIND REASONING** (Theory of Mind variants only)
- P2 now knows their Red cards and can make informed decisions
- Clue giver likely thinks I need to understand P2's new Red knowledge
- This may affect coordination around Red fireworks

**PART 4: STRATEGIC IMPLICATIONS**
- P2 may be able to play Red cards if fireworks align
- Team focus might shift toward Red foundation building
- Future clues can build on this Red information

```json
{
  "updated_belief_graph": "..."
}
```
"""
        return cot_prompt
    
    def extract_reasoning_and_action(self, response: str) -> tuple[str, str]:
        """Extract reasoning and action from LLM response."""
        response = response.strip()
        
        # Look for REASONING section
        reasoning_start = response.find("REASONING:")
        if reasoning_start == -1:
            # Try alternative patterns
            reasoning_start = response.find("**REASONING:**")
            if reasoning_start == -1:
                reasoning_start = 0
        
        # Find where JSON starts
        json_start = response.find("{")
        
        if json_start == -1:
            # No JSON found
            return response, ""
        
        if reasoning_start < json_start:
            reasoning = response[reasoning_start:json_start].strip()
            action_json = response[json_start:].strip()
        else:
            reasoning = "No explicit reasoning provided."
            action_json = response[json_start:].strip()
        
        # Clean up reasoning
        if reasoning.startswith("REASONING:") or reasoning.startswith("**REASONING:**"):
            reasoning = reasoning.split(":", 1)[1].strip()
        
        return reasoning, action_json

    def extract_belief_update_reasoning_and_json(self, response: str) -> tuple[str, str]:
        """Extract reasoning and JSON belief graph from LLM belief update response."""
        response = response.strip()

        # Find JSON block first
        json_start = response.find("```json")
        if json_start == -1:
            json_start = response.find("```")

        if json_start == -1:
            # No JSON block found, look for first {
            json_start = response.find("{")

        if json_start == -1:
            # No JSON found
            return response, ""

        # Find the END of the JSON block (the closing ```)
        json_end = response.find("```", json_start + 3)
        if json_end == -1:
            # No closing ```, take everything from json_start
            json_section = response[json_start:].strip()
            reasoning = "No explicit reasoning provided."
        else:
            # Extract just the JSON block content
            json_section = response[json_start:json_end].strip()

            # Look for reasoning AFTER the JSON block
            reasoning_after_json = response.find("REASONING:", json_end)
            if reasoning_after_json == -1:
                reasoning_after_json = response.find("**REASONING:**", json_end)

            if reasoning_after_json != -1:
                reasoning = response[reasoning_after_json:].strip()
            else:
                # Try reasoning BEFORE JSON
                reasoning_start = response.find("REASONING:")
                if reasoning_start == -1:
                    reasoning_start = response.find("**REASONING:**")

                if reasoning_start != -1 and reasoning_start < json_start:
                    reasoning = response[reasoning_start:json_start].strip()
                else:
                    reasoning = "No explicit reasoning provided."

        # Extract JSON from code block
        if json_section.startswith("```json"):
            json_section = json_section[7:]  # Remove ```json
        if json_section.startswith("```"):
            lines = json_section.split("\n")
            if len(lines) > 1:
                json_section = "\n".join(lines[1:])  # Remove first ```
                # Remove last line if it's just ```
                if json_section.endswith("```"):
                    json_section = json_section[:-3].strip()
            else:
                json_section = lines[0].replace("```", "")
        json_section = json_section.strip()

        # Clean up reasoning
        if reasoning.startswith("REASONING:") or reasoning.startswith("**REASONING:**"):
            reasoning = reasoning.split(":", 1)[1].strip()

        return reasoning, json_section
    
    def validate_reasoning_quality(self, reasoning: str) -> Dict[str, bool]:
        """Validate the quality of reasoning provided."""
        quality_checks = {
            "has_situation_analysis": any(
                keyword in reasoning.lower()
                for keyword in ["situation", "game state", "current state", "tokens", "fireworks"]
            ),
            "has_legal_moves_analysis": any(
                keyword in reasoning.lower()
                for keyword in ["legal moves", "available", "options", "can play", "can hint"]
            ),
            "has_strategic_evaluation": any(
                keyword in reasoning.lower()
                for keyword in ["strategy", "risk", "benefit", "priority", "best", "optimal"]
            ),
            "has_teammate_consideration": any(
                keyword in reasoning.lower()
                for keyword in ["teammate", "partner", "their hand", "they have", "help them"]
            ),
            "has_belief_graph_influence": any(
                keyword in reasoning.lower()
                for keyword in ["belief graph", "belief", "confidence", "team focus", "certain", "probability", "tom", "theory of mind"]
            ),
            "has_final_decision": any(
                keyword in reasoning.lower()
                for keyword in ["decision", "choose", "because", "therefore", "so i will"]
            ),
            "sufficient_length": len(reasoning.split()) >= 20
        }

        return quality_checks
    
    def score_reasoning_quality(self, reasoning: str) -> float:
        """Score the quality of reasoning from 0.0 to 1.0."""
        quality_checks = self.validate_reasoning_quality(reasoning)
        return sum(quality_checks.values()) / len(quality_checks)
    
    def create_reasoning_feedback(self, reasoning: str) -> str:
        """Create feedback on reasoning quality for retry attempts."""
        quality_checks = self.validate_reasoning_quality(reasoning)
        missing_elements = [
            element.replace("has_", "").replace("_", " ").title()
            for element, present in quality_checks.items() 
            if not present
        ]
        
        if not missing_elements:
            return "Reasoning quality is good."
        
        feedback = "Your reasoning is missing these elements: " + ", ".join(missing_elements)
        feedback += "\n\nPlease provide more detailed reasoning covering all aspects of your decision-making process."
        
        return feedback

    def validate_belief_update_reasoning_quality(self, reasoning: str) -> Dict[str, bool]:
        """Validate the quality of belief update reasoning provided."""
        quality_checks = {
            "has_direct_impact_analysis": any(
                keyword in reasoning.lower()
                for keyword in ["direct impact", "specific information", "cards received", "100% certain", "negative inference"]
            ),
            "has_team_mental_model_updates": any(
                keyword in reasoning.lower()
                for keyword in ["team strategy", "team priorities", "clue giver", "intent", "assessment", "decision-making"]
            ),
            "has_theory_of_mind_reasoning": any(
                keyword in reasoning.lower()
                for keyword in ["theory of mind", "recipient's knowledge", "recursive", "coordination", "shared understanding"]
            ),
            "has_strategic_implications": any(
                keyword in reasoning.lower()
                for keyword in ["strategic implications", "future actions", "risk assessment", "coordination opportunities"]
            ),
            "has_belief_graph_analysis": any(
                keyword in reasoning.lower()
                for keyword in ["belief", "update", "knowledge landscape", "information", "clue"]
            ),
            "has_structured_reasoning": any(
                pattern in reasoning.lower()
                for pattern in ["part 1", "part 2", "part 3", "part 4"]
            ),
            "sufficient_length": len(reasoning.split()) >= 30
        }

        return quality_checks

    def score_belief_update_reasoning_quality(self, reasoning: str) -> float:
        """Score the quality of belief update reasoning from 0.0 to 1.0."""
        quality_checks = self.validate_belief_update_reasoning_quality(reasoning)
        return sum(quality_checks.values()) / len(quality_checks)

    def create_belief_update_reasoning_feedback(self, reasoning: str) -> str:
        """Create feedback on belief update reasoning quality for retry attempts."""
        quality_checks = self.validate_belief_update_reasoning_quality(reasoning)
        missing_elements = [
            element.replace("has_", "").replace("_", " ").title()
            for element, present in quality_checks.items()
            if not present
        ]

        if not missing_elements:
            return "Belief update reasoning quality is good."

        feedback = "Your belief update reasoning should include these structured elements: " + ", ".join(missing_elements)
        feedback += "\n\nPlease provide detailed Chain of Thought analysis following the 4-part structure:\n"
        feedback += "1. Direct Impact Analysis\n"
        feedback += "2. Team Mental Model Updates\n"
        feedback += "3. Theory of Mind Reasoning (if applicable)\n"
        feedback += "4. Strategic Implications"

        return feedback