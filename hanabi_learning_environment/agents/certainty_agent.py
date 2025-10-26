"""Belief graph certainty agent - Tracks possible values as sets.

Variation 1: The Certainty Graph
Tracks possible values as sets (certainty-based).
"""

import json
from typing import Dict, Any, Optional

from ..agents.belief_graph_base import BeliefGraphAgentBase


class BeliefGraphCertaintyAgent(BeliefGraphAgentBase):
    """Variation 1: The Certainty Graph

    Tracks possible values as sets (certainty-based).
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the certainty belief graph agent."""
        super().__init__(config, 'certainty', 'BG_certainty', *args, **kwargs)

    def _initialize_variant_specific_beliefs(self):
        """Initialize certainty-specific belief structure."""
        pass

    def _create_belief_entry(self) -> Dict[str, Any]:
        """Create a certainty-based belief entry."""
        return {
            "possible_colors": ["red", "blue", "green", "white", "yellow"],
            "possible_ranks": [1, 2, 3, 4, 5]
        }

    def _process_single_event(self, event: Dict[str, Any], observation: Dict[str, Any]):
        """Process a single event through the LLM."""
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        self.logger.log_info("BELIEF_BEFORE_UPDATE", f"Current belief graph: {json.dumps(self.belief_graph, indent=2)}")

        # Create history like in gemini_agent.py
        history = self.prompt_manager.format_history_for_llm(self.observation_history, self.action_history)

        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'certainty', history, observation)

        self.logger.log_info("PROMPT_TO_LLM", f"Prompt ({len(prompt)} chars):\n{prompt}")

        try:
            reasoning_response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2}
            )

            self.logger.log_info("LLM_OUTPUT", f"LLM Response:\n{reasoning_response.text}")

            response_text = reasoning_response.text
            json_text = self._extract_json_from_response(response_text)

            self.logger.log_debug("BELIEF_UPDATE_JSON_EXTRACTED", json_text)

            new_belief_graph = json.loads(json_text)

            diff_summary = self._generate_belief_diff(self.belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            self.belief_graph = new_belief_graph

            # CRITICAL: Update snapshots after LLM belief update to keep them in sync
            self._update_snapshots_after_belief_update()

            self.logger.log_info("BELIEF_AFTER_UPDATE", f"Updated belief graph:\n{json.dumps(self.belief_graph, indent=2)}")

        except Exception as e:
            self.logger.log_error("UPDATE_ERROR", f"Failed to update beliefs: {e}")