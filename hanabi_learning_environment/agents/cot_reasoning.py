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
1. **SITUATION ANALYSIS**: What is the current game state? What are the key constraints?
2. **CARD KNOWLEDGE CHECK**: Do I know any 1s I can play? Do I know any other playable cards?
3. **TEAMMATE SCAN**: What 1s or immediately playable cards do I see in teammates' hands?
4. **LEGAL MOVES ANALYSIS**: What moves are available to me right now?
5. **STRATEGIC EVALUATION**: What would each legal move accomplish? What are the risks/benefits?
6. **PRIORITY ASSESSMENT**: 1s first, then other safe plays, then helpful hints, then safe discards
7. **FINAL DECISION**: Which action serves our team's goal best and why?

**FORMAT:**
First provide your detailed reasoning in the REASONING section above.
Then provide your action as a JSON object.

Example:
REASONING:
1. SITUATION ANALYSIS: We have 7 information tokens and 3 life tokens. Fireworks progress is slow with only Red at level 1.
2. LEGAL MOVES ANALYSIS: I can play 5 cards, discard 5 cards, or give 8 different hints.
3. STRATEGIC EVALUATION: Playing without certainty risks life tokens. Discarding wastes cards. Hinting helps teammate.
4. TEAMMATE CONSIDERATION: I see my teammate has a Red 2 that could be played next if they know about it.
5. PRIORITY ASSESSMENT: Information sharing is crucial right now to enable plays.
6. FINAL DECISION: I'll hint about the Red card to enable an immediate play.

{"action_type": "REVEAL_COLOR", "color": "R", "target_offset": 1}
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