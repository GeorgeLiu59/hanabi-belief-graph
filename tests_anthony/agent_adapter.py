"""
Adapter to use GeminiAgent with ablation study infrastructure.

This adapter converts ablation test scenarios into observations that GeminiAgent
can process, and extracts both actions and reasoning for analysis.
"""

from typing import Dict, Any, Tuple
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hanabi_learning_environment.agents.gemini_agent import GeminiAgent
from hanabi_learning_environment.agents.probabilistic_agent import BeliefGraphProbabilisticAgent
from hanabi_learning_environment.agents.cot_reasoning import ChainOfThoughtReasoner


class AblationAgentAdapter:
    """
    Adapter to use existing Hanabi agents with ablation study tests.

    Converts ablation test scenarios (from ablation_scenarios.json) into
    agent observations, and captures both actions and reasoning.
    """

    def __init__(self, agent_type: str = "probabilistic", model: str = "gemini-2.0-flash-lite"):
        """
        Initialize agent for ablation testing.

        Args:
            agent_type: "probabilistic", "certainty", or "theory_of_mind"
            model: Gemini model to use
        """
        self.agent_type = agent_type

        # Initialize agent based on type
        config = {
            "players": 2,
            "model": model,
            "log_dir": "logs/ablation_study"
        }

        # Use appropriate agent variant
        if agent_type == "probabilistic":
            self.agent = BeliefGraphProbabilisticAgent(config)
        else:
            # For now, default to probabilistic
            # TODO: Add other variants when needed
            self.agent = BeliefGraphProbabilisticAgent(config)

        # Initialize agent properly
        self.agent.set_episode(1)
        self.agent.reset(config)

        self.cot_reasoner = ChainOfThoughtReasoner()

        # Track last reasoning for extraction
        self.last_reasoning = ""

    def decide_with_custom_context(
        self,
        scenario: Dict[str, Any],
        custom_context: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Make decision with custom context (for ablation study).

        This bypasses the normal belief graph generation and uses the
        provided context instead (which could be raw game state, minimal
        context, natural language beliefs, etc.).

        Args:
            scenario: Test scenario from ablation_scenarios.json
            custom_context: Pre-generated context string (from context_generators.py)

        Returns:
            (action_dict, reasoning_text)
        """
        # Convert scenario to observation format
        observation = self._scenario_to_observation(scenario)

        # TODO: Modify agent to accept custom context
        # For now, use the standard agent.act_with_reasoning() method
        # which uses the agent's built-in belief graph

        # Call agent with reasoning extraction
        action, reasoning = self.agent.act_with_reasoning(observation)

        # Convert action format from agent format to test format
        # Agent returns: {"action_type": "DISCARD", "card_index": 0}
        # Tests expect: {"type": "DISCARD", "target": 0}
        if action:
            target = action.get("card_index")
            if target is None:
                target = action.get("target_offset")

            action = {
                "type": action.get("action_type"),
                "target": target,
                "color": action.get("color"),
                "rank": action.get("rank")
            }

        return action, reasoning

    def _scenario_to_observation(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert ablation test scenario to agent observation format.

        Args:
            scenario: Dict from ablation_scenarios.json

        Returns:
            Observation dict in Hanabi environment format
        """
        # Build observation in the format your agent expects
        observation = {
            "current_player": scenario["current_player"],
            "current_player_offset": 0,  # Always 0 for agent's perspective
            "life_tokens": scenario["life_tokens"],
            "information_tokens": scenario["info_tokens"],
            "num_players": 2,
            "deck_size": scenario["deck_size"],

            # Fireworks
            "fireworks": scenario["fireworks"],

            # Discard pile
            "discard_pile": scenario["discard_pile"],

            # Observed hands (other players' visible cards)
            "observed_hands": [
                scenario["player_hands"]["player_1"]  # Player 1's hand (visible to agent)
            ],

            # Card knowledge - nested list: card_knowledge[player_offset][card_index]
            # card_knowledge[0] is the agent's own hand knowledge
            "card_knowledge": [
                [
                    {"color": None, "rank": None}  # Agent doesn't know their own cards
                    for _ in scenario["player_hands"]["player_0"]
                ]
            ],

            # Legal moves - generate basic legal moves
            "legal_moves": [
                {"action_type": "DISCARD", "card_index": i}
                for i in range(len(scenario["player_hands"]["player_0"]))
            ] + [
                {"action_type": "PLAY", "card_index": i}
                for i in range(len(scenario["player_hands"]["player_0"]))
            ],

            # Additional required fields
            "last_moves": [],  # No previous moves for first turn
        }

        return observation

    def _extract_reasoning_from_logs(self) -> str:
        """
        Extract reasoning from agent's log files.

        Returns:
            Reasoning text string
        """
        # TODO: Implement log extraction or modify agent to return reasoning
        # For now, return placeholder
        return "Reasoning not yet extracted from agent logs. TODO: Implement log parsing or modify agent to return reasoning directly."


def create_agent_decision_function(agent_type: str = "probabilistic"):
    """
    Factory function to create agent decision function for ablation tests.

    Args:
        agent_type: Type of belief graph agent to use

    Returns:
        Function matching the signature expected by ablation tests:
        (scenario, context) -> (action, reasoning)
    """
    adapter = AblationAgentAdapter(agent_type=agent_type)

    def decide(scenario: Dict[str, Any], context: str) -> Tuple[Dict[str, Any], str]:
        """
        Decision function matching ablation test signature.

        Args:
            scenario: Test scenario dict
            context: Pre-generated context string

        Returns:
            (action_dict, reasoning_text)
        """
        return adapter.decide_with_custom_context(scenario, context)

    return decide
