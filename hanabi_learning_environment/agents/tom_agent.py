"""Belief graph Theory of Mind agent - Adds player modeling and team strategy.

Variation 3: The "Player Read" Graph (Recursive ToM)
Adds Theory of Mind layer on top of probabilistic beliefs.
"""

import json
from typing import Dict, Any

from ..agents.belief_graph_base import BeliefGraphAgentBase


class BeliefGraphToMAgent(BeliefGraphAgentBase):
    """Variation 3: The "Player Read" Graph (Recursive ToM)

    Adds Theory of Mind layer on top of probabilistic beliefs.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the ToM belief graph agent."""
        super().__init__(config, 'theory_of_mind', 'BG_tom', *args, **kwargs)

    def _initialize_variant_specific_beliefs(self):
        """Initialize ToM-specific belief structure."""
        self.uniform_color = {"red": 0.2, "blue": 0.2, "green": 0.2, "white": 0.2, "yellow": 0.2}
        self.uniform_rank = {"1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2, "5": 0.2}

        self.belief_graph["ToM_Layer"] = {
            "Teammates": {},
            "Team_Focus": {
                "focus_distribution": {
                    "PLAY_ANYTHING": 0.9,
                    "SAVE_FIVES": 0.05,
                    "PLAY_ONES": 0.05
                }
            }
        }

    def _create_belief_entry(self) -> Dict[str, Any]:
        """Create a ToM probabilistic belief entry."""
        return {
            "color_distribution": self.uniform_color.copy(),
            "rank_distribution": self.uniform_rank.copy()
        }

    def _initialize_variant_specific_teammate_models(self, observation: Dict[str, Any]):
        """Initialize ToM teammate models."""
        num_players = observation['num_players']
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                self.belief_graph["ToM_Layer"]["Teammates"][f"P{player_id}"] = {
                    "inferred_skill": 0.5,
                    "play_aggressiveness": 0.5,
                    "hint_quality": 0.5
                }

    def _process_single_event(self, event: Dict[str, Any]):
        """Process a single event through the LLM with ToM reasoning."""
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'theory_of_mind')

        self.logger.log_info("TOM_UPDATE_PROMPT", f"Sending ToM update to LLM: {len(prompt)} chars")
        self.logger.log_debug("TOM_UPDATE_PROMPT_DETAIL", prompt)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'response_mime_type': 'application/json', 'temperature': 0.2}
            )

            self.logger.log_info("TOM_UPDATE_RESPONSE", f"LLM response: {len(response.text)} chars")
            self.logger.log_debug("TOM_UPDATE_RESPONSE_DETAIL", response.text)

            previous_belief_graph = self.belief_graph.copy()

            new_belief_graph = json.loads(response.text)

            diff_summary = self._generate_belief_diff(previous_belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            self.belief_graph = new_belief_graph

            if 'ToM_Layer' in self.belief_graph:
                self.logger.log_info("TOM_UPDATE", f"Team focus: {self.belief_graph['ToM_Layer']['Team_Focus']}")
                self.logger.log_info("LLM_UPDATE", "ToM belief graph updated via LLM")

        except Exception as e:
            self.logger.log_error("TOM_ERROR", f"Failed to update ToM beliefs: {e}")

    def _generate_variant_specific_diff(self, before: Dict, after: Dict) -> str:
        """Generate ToM-specific diff for Team_Focus and teammate skills."""
        diff_summary = ""

        if 'ToM_Layer' in before and 'ToM_Layer' in after:
            before_tom = before['ToM_Layer']
            after_tom = after['ToM_Layer']

            if 'Team_Focus' in before_tom and 'Team_Focus' in after_tom:
                before_focus = before_tom['Team_Focus']['focus_distribution']
                after_focus = after_tom['Team_Focus']['focus_distribution']

                for focus_type in before_focus:
                    if before_focus[focus_type] != after_focus[focus_type]:
                        diff_summary += f"  Team_Focus.{focus_type}: {before_focus[focus_type]:.2f} → {after_focus[focus_type]:.2f}\n"

            if 'Teammates' in before_tom and 'Teammates' in after_tom:
                for player in before_tom['Teammates']:
                    if player in after_tom['Teammates']:
                        before_player = before_tom['Teammates'][player]
                        after_player = after_tom['Teammates'][player]

                        for attribute in before_player:
                            if before_player[attribute] != after_player[attribute]:
                                diff_summary += f"  {player}.{attribute}: {before_player[attribute]:.2f} → {after_player[attribute]:.2f}\n"

        return diff_summary