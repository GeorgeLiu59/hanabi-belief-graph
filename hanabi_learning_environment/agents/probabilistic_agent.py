"""Belief graph probabilistic agent - Tracks probability distributions.

Variation 2: The Belief Graph (Probabilistic)
Tracks probability distributions instead of sets.
"""

import json
from typing import Dict, Any

from ..agents.belief_graph_base import BeliefGraphAgentBase


class BeliefGraphProbabilisticAgent(BeliefGraphAgentBase):
    """Variation 2: The Belief Graph (Probabilistic)

    Tracks probability distributions instead of sets.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the probabilistic belief graph agent."""
        super().__init__(config, 'probabilistic', 'BG_probabilistic', *args, **kwargs)

    def _initialize_variant_specific_beliefs(self):
        """Initialize probabilistic-specific belief structure."""
        self.uniform_color = {"red": 0.2, "blue": 0.2, "green": 0.2, "white": 0.2, "yellow": 0.2}
        self.uniform_rank = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}

    def _create_belief_entry(self) -> Dict[str, Any]:
        """Create a probabilistic belief entry."""
        return {
            "color_distribution": self.uniform_color.copy(),
            "rank_distribution": self.uniform_rank.copy()
        }

    def _process_single_event(self, event: Dict[str, Any], observation: Dict[str, Any]):
        """Process a single event through the LLM for probabilistic beliefs."""
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        # Create history like in gemini_agent.py
        history = self.prompt_manager.format_history_for_llm(self.observation_history, self.action_history)

        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'probabilistic', history, observation)

        self.logger.log_info("BELIEF_UPDATE_PROMPT", f"Sending probabilistic update to LLM: {len(prompt)} chars")
        self.logger.log_debug("BELIEF_UPDATE_PROMPT_DETAIL", prompt)

        try:
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2}
            )

            self.logger.log_info("BELIEF_UPDATE_RESPONSE", f"LLM reasoning response: {len(reasoning_response.text)} chars")
            self.logger.log_debug("BELIEF_UPDATE_COT_REASONING", reasoning_response.text)

            response_text = reasoning_response.text

            json_text = self._extract_json_from_response(response_text)
            self.logger.log_debug("BELIEF_UPDATE_JSON_EXTRACTED", json_text)

            previous_belief = json.dumps(self.belief_graph, indent=2)
            self.logger.log_debug("BELIEF_UPDATE_BEFORE", previous_belief)

            new_belief_graph = json.loads(json_text)

            diff_summary = self._generate_belief_diff(self.belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            self.belief_graph = new_belief_graph

            # CRITICAL: Update snapshots after LLM belief update to keep them in sync
            self._update_snapshots_after_belief_update()

            updated_belief = json.dumps(self.belief_graph, indent=2)
            self.logger.log_debug("BELIEF_UPDATE_AFTER", updated_belief)
            self.logger.log_info("LLM_UPDATE", "Probabilistic belief graph updated via LLM")

        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update probabilistic beliefs: {e}")