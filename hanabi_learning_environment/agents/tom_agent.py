"""Belief graph Theory of Mind agent - Adds player modeling and team strategy.

Variation 3: The "Player Read" Graph (Recursive ToM)
Adds Theory of Mind layer on top of probabilistic beliefs.
"""

import json
from typing import Dict, Any

from ..agents.probabilistic_agent import BeliefGraphProbabilisticAgent
from ..agents.agent_logger import AgentLogger
from ..agents.cot_reasoning import ChainOfThoughtReasoner


class BeliefGraphToMAgent(BeliefGraphProbabilisticAgent):
    """Variation 4: The "Player Read" Graph (Recursive ToM)

    Adds Theory of Mind layer on top of probabilistic beliefs.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the ToM belief graph agent."""
        super().__init__(config, *args, **kwargs)
        # Override the agent_id and variant to be TOM-specific
        self.variant = 'theory_of_mind'
        self.agent_id = f"BG_tom_{self.agent_id.split('_')[-1]}"
        self.logger = AgentLogger(self.agent_id)
        # Initialize CoT reasoner
        self.cot_reasoner = ChainOfThoughtReasoner()

    def _initialize_variant_specific_beliefs(self):
        """Initialize ToM-specific belief structure with recursive reasoning."""
        # Call parent to initialize probabilistic beliefs
        super()._initialize_variant_specific_beliefs()

        # Add ToM layer on top
        self.belief_graph["ToM_Layer"] = {
            "Teammates": {},
            "Recursive_ToM": {},  # New: Second-order beliefs about what others think
            "Team_Focus": {
                "focus_distribution": {
                    "PLAY_ANYTHING": 0.9,
                    "SAVE_FIVES": 0.05,
                    "PLAY_ONES": 0.05
                }
            }
        }

    def _initialize_variant_specific_teammate_models(self, observation: Dict[str, Any]):
        """Initialize ToM teammate models with recursive reasoning per specification."""
        num_players = observation['num_players']
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                player_key = f"P{player_id}"

                # First-order ToM: What I think about this teammate
                self.belief_graph["ToM_Layer"]["Teammates"][player_key] = {
                    "inferred_skill": 0.5,
                    "play_aggressiveness": 0.5,
                    "hint_quality": 0.5
                }

                # Second-order ToM (Variation 4): What I think this teammate thinks about ME (player n itself)
                my_key = f"P{self.my_player_number + 1}"
                self.belief_graph["ToM_Layer"]["Recursive_ToM"][player_key] = {
                    my_key: {
                        "believes_about_my_hand": {},  # What this player thinks about my hand beliefs
                        "believes_about_my_skill": 0.5,  # What this player thinks about my skill
                        "believes_about_my_strategy": 0.5  # What this player thinks about my play style
                    }
                }

    def _update_variant_specific_models(self, observation: Dict[str, Any]):
        """Update ToM teammate models - ensure all current players are tracked with recursive beliefs."""
        num_players = observation['num_players']

        # Initialize ToM layer if it doesn't exist
        if "ToM_Layer" not in self.belief_graph:
            self.belief_graph["ToM_Layer"] = {
                "Team_Focus": {
                    "focus_distribution": {
                        "building_fireworks": 0.4,
                        "saving_cards": 0.3,
                        "information_management": 0.3
                    }
                },
                "Teammates": {},
                "Recursive_ToM": {}
            }

        # Ensure all current players have models
        for player_idx in range(num_players):
            if player_idx != self.my_player_number:
                player_id = player_idx + 1
                player_key = f"P{player_id}"
                my_key = f"P{self.my_player_number + 1}"

                # First-order ToM
                if player_key not in self.belief_graph["ToM_Layer"]["Teammates"]:
                    self.belief_graph["ToM_Layer"]["Teammates"][player_key] = {
                        "inferred_skill": 0.5,
                        "play_aggressiveness": 0.5,
                        "hint_quality": 0.5
                    }

                # Second-order ToM (Variation 4): What this player thinks about ME
                if player_key not in self.belief_graph["ToM_Layer"]["Recursive_ToM"]:
                    self.belief_graph["ToM_Layer"]["Recursive_ToM"][player_key] = {
                        my_key: {
                            "believes_about_my_hand": {},
                            "believes_about_my_skill": 0.5,
                            "believes_about_my_strategy": 0.5
                        }
                    }

        # Remove players who are no longer in the game (shouldn't happen but just in case)
        current_players = {f"P{i+1}" for i in range(num_players) if i != self.my_player_number}
        existing_players = set(self.belief_graph["ToM_Layer"]["Teammates"].keys())
        players_to_remove = existing_players - current_players
        for player_key in players_to_remove:
            del self.belief_graph["ToM_Layer"]["Teammates"][player_key]
            if player_key in self.belief_graph["ToM_Layer"]["Recursive_ToM"]:
                del self.belief_graph["ToM_Layer"]["Recursive_ToM"][player_key]

    def _process_single_event(self, event: Dict[str, Any], observation: Dict[str, Any]):
        """Process a single event through the LLM with ToM reasoning and detailed CoT logging."""
        if 'clue_recipient' not in event:
            self.logger.log_debug("BELIEF_UPDATE", "No belief update needed for non-clue events")
            return

        self.logger.log_info("TOM_BELIEF_UPDATE_START", f"=== ToM BELIEF UPDATE START ===")
        self.logger.log_debug("TOM_EVENT_DETAILS", f"Processing event: {event}")

        # Log detailed ToM reasoning analysis
        self._log_tom_reasoning_analysis(event, observation)

        # Create history like in gemini_agent.py
        history = self.prompt_manager.format_history_for_llm(self.observation_history, self.action_history)

        prompt = self.prompt_manager.get_belief_update_prompt(event, self.belief_graph, 'theory_of_mind', history, observation)

        self.logger.log_info("TOM_UPDATE_PROMPT", f"Sending ToM update to LLM: {len(prompt)} chars")
        self.logger.log_debug("TOM_UPDATE_PROMPT_DETAIL", prompt)

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={'temperature': 0.2}
            )

            self.logger.log_info("TOM_UPDATE_RESPONSE", f"LLM response: {len(response.text)} chars")
            self.logger.log_debug("TOM_UPDATE_RESPONSE_DETAIL", response.text)

            previous_belief_graph = self.belief_graph.copy()

            response_text = response.text

            # Extract CoT reasoning and JSON using the CoT helper
            reasoning, json_text = self.cot_reasoner.extract_belief_update_reasoning_and_json(response_text)

            # Log the CoT reasoning
            if reasoning:
                self.logger.log_info("TOM_COT_REASONING", f"Chain of Thought reasoning:\n{reasoning}")
                self.logger.log_info("TOM_COT_SUCCESS", "✓ CoT reasoning extracted successfully")
            else:
                self.logger.log_warning("TOM_COT_MISSING", "⚠ No CoT reasoning found in response")

            self.logger.log_debug("TOM_UPDATE_JSON_EXTRACTED", json_text)
            self.logger.log_debug("TOM_JSON_LENGTH", f"JSON text length: {len(json_text)} chars")

            try:
                new_belief_graph = json.loads(json_text)
                self.logger.log_debug("TOM_JSON_PARSE_SUCCESS", "✓ JSON parsed successfully")
            except json.JSONDecodeError as e:
                self.logger.log_error("TOM_JSON_PARSE_ERROR", f"JSON parsing failed: {e}")
                self.logger.log_error("TOM_JSON_PARSE_ERROR", f"JSON text was: {json_text[:500]}...")
                raise

            # Log detailed ToM layer changes with reasoning
            self._log_tom_layer_changes(previous_belief_graph, new_belief_graph, event)

            diff_summary = self._generate_belief_diff(previous_belief_graph, new_belief_graph)
            self.logger.log_info("BELIEF_UPDATE_DIFF", diff_summary)

            self.belief_graph = new_belief_graph

            # CRITICAL: Update snapshots after LLM belief update to keep them in sync
            self._update_snapshots_after_belief_update()

            if 'ToM_Layer' in self.belief_graph:
                self.logger.log_info("TOM_UPDATE", f"Team focus: {self.belief_graph['ToM_Layer']['Team_Focus']}")
                self.logger.log_info("LLM_UPDATE", "ToM belief graph updated via LLM")
                self.logger.log_info("TOM_BELIEF_UPDATE_COMPLETE", f"=== ToM BELIEF UPDATE COMPLETE ===")
            else:
                self.logger.log_warning("TOM_MISSING_LAYER", "⚠ No ToM_Layer found in updated belief graph")

        except Exception as e:
            self.logger.log_error("TOM_ERROR", f"Failed to update ToM beliefs: {e}")

    def _log_tom_reasoning_analysis(self, event: Dict[str, Any], observation: Dict[str, Any]):
        """Log detailed Chain of Thought reasoning for ToM updates."""
        clue_giver = event.get('clue_giver', 'Unknown')
        clue_recipient = event.get('clue_recipient', 'Unknown')
        clue_type = event.get('clue_type', 'Unknown')
        clue_value = event.get('value', 'Unknown')

        self.logger.log_info("TOM_COT_ANALYSIS", f"=== THEORY OF MIND REASONING FOR {clue_giver}→{clue_recipient} ({clue_type}: {clue_value}) ===")

        # First-order ToM reasoning
        self.logger.log_info("TOM_COT_FIRST_ORDER", f"FIRST-ORDER ANALYSIS: What I think about {clue_recipient}'s updated beliefs")
        self.logger.log_info("TOM_COT_FIRST_ORDER", f"• {clue_recipient} now knows their hinted card(s) are {clue_value}")
        self.logger.log_info("TOM_COT_FIRST_ORDER", f"• {clue_recipient} knows their other cards are NOT {clue_value} (negative inference)")
        self.logger.log_info("TOM_COT_FIRST_ORDER", f"• This may change {clue_recipient}'s playable options and decision-making")

        # Second-order ToM reasoning (Variation 4)
        self.logger.log_info("TOM_COT_SECOND_ORDER", f"SECOND-ORDER ANALYSIS: What I think {clue_giver} thinks about ME")
        self.logger.log_info("TOM_COT_SECOND_ORDER", f"• {clue_giver} may be signaling information about MY hand or strategy")
        self.logger.log_info("TOM_COT_SECOND_ORDER", f"• {clue_giver} might think I need this information for upcoming decisions")
        self.logger.log_info("TOM_COT_SECOND_ORDER", f"• This could affect what {clue_giver} thinks about my skill level")

        # Strategic implications
        self.logger.log_info("TOM_COT_STRATEGIC", f"STRATEGIC IMPLICATIONS: Team dynamics and coordination")
        self.logger.log_info("TOM_COT_STRATEGIC", f"• Team focus may shift based on this information exchange")
        self.logger.log_info("TOM_COT_STRATEGIC", f"• Player skill assessments may update based on clue quality/timing")
        self.logger.log_info("TOM_COT_STRATEGIC", f"• This affects our shared mental model and future coordination")

    def _log_tom_layer_changes(self, before: Dict, after: Dict, event: Dict[str, Any]):
        """Log detailed ToM layer changes with CoT justification."""
        if 'ToM_Layer' not in before or 'ToM_Layer' not in after:
            return

        before_tom = before['ToM_Layer']
        after_tom = after['ToM_Layer']
        clue_giver = event.get('clue_giver', 'Unknown')

        self.logger.log_info("TOM_LAYER_UPDATE", "=== ToM LAYER CHANGES WITH JUSTIFICATION ===")

        # Team Focus changes
        if 'Team_Focus' in before_tom and 'Team_Focus' in after_tom:
            before_focus = before_tom['Team_Focus']['focus_distribution']
            after_focus = after_tom['Team_Focus']['focus_distribution']

            for focus_type, before_prob in before_focus.items():
                after_prob = after_focus.get(focus_type, before_prob)
                if abs(before_prob - after_prob) > 0.01:
                    direction = "↑" if after_prob > before_prob else "↓"
                    self.logger.log_info("TOM_TEAM_FOCUS_UPDATE", f"Team focus {direction}: {focus_type} {before_prob:.2f} → {after_prob:.2f}")
                    self.logger.log_info("TOM_TEAM_FOCUS_REASONING", f"Because {clue_giver}'s action suggests team is prioritizing {focus_type}")

        # First-order ToM changes (teammate skills)
        if 'Teammates' in before_tom and 'Teammates' in after_tom:
            for player in before_tom['Teammates']:
                if player in after_tom['Teammates']:
                    before_player = before_tom['Teammates'][player]
                    after_player = after_tom['Teammates'][player]

                    for attribute, before_val in before_player.items():
                        after_val = after_player.get(attribute, before_val)
                        if abs(before_val - after_val) > 0.01:
                            direction = "↑" if after_val > before_val else "↓"
                            self.logger.log_info("TOM_TEAMMATE_UPDATE", f"{player} {attribute} {direction}: {before_val:.2f} → {after_val:.2f}")
                            self.logger.log_info("TOM_TEAMMATE_REASONING", f"Because {clue_giver}'s clue shows {player} has {'higher' if after_val > before_val else 'lower'} {attribute}")

        # Second-order ToM changes (what others think about ME)
        if 'Recursive_ToM' in before_tom and 'Recursive_ToM' in after_tom:
            for player_key in before_tom['Recursive_ToM']:
                if player_key in after_tom['Recursive_ToM']:
                    before_recursive = before_tom['Recursive_ToM'][player_key]
                    after_recursive = after_tom['Recursive_ToM'][player_key]

                    for me_player in before_recursive:
                        if me_player in after_recursive:
                            before_beliefs = before_recursive[me_player]
                            after_beliefs = after_recursive[me_player]

                            # Skill assessment changes
                            if 'believes_about_my_skill' in before_beliefs and 'believes_about_my_skill' in after_beliefs:
                                before_skill = before_beliefs['believes_about_my_skill']
                                after_skill = after_beliefs['believes_about_my_skill']
                                if abs(before_skill - after_skill) > 0.01:
                                    direction = "↑" if after_skill > before_skill else "↓"
                                    self.logger.log_info("TOM_RECURSIVE_UPDATE", f"{player_key} thinks my skill {direction}: {before_skill:.2f} → {after_skill:.2f}")
                                    self.logger.log_info("TOM_RECURSIVE_REASONING", f"Because {clue_giver}'s action to {player_key} suggests they {'respect' if after_skill > before_skill else 'doubt'} my capabilities")

                            # Strategy assessment changes
                            if 'believes_about_my_strategy' in before_beliefs and 'believes_about_my_strategy' in after_beliefs:
                                before_strategy = before_beliefs['believes_about_my_strategy']
                                after_strategy = after_beliefs['believes_about_my_strategy']
                                if abs(before_strategy - after_strategy) > 0.01:
                                    direction = "↑" if after_strategy > before_strategy else "↓"
                                    self.logger.log_info("TOM_RECURSIVE_UPDATE", f"{player_key} thinks my strategy {direction}: {before_strategy:.2f} → {after_strategy:.2f}")
                                    self.logger.log_info("TOM_RECURSIVE_REASONING", f"Because {clue_giver}'s behavior indicates they think I'm playing {'more aggressively' if after_strategy > before_strategy else 'more conservatively'}")

        self.logger.log_info("TOM_LAYER_UPDATE", "=== END ToM LAYER UPDATES ===")

    def _generate_variant_specific_diff(self, before: Dict, after: Dict) -> str:
        """Generate ToM-specific diff for Team_Focus, teammate skills, and recursive beliefs."""
        diff_summary = ""

        if 'ToM_Layer' in before and 'ToM_Layer' in after:
            before_tom = before['ToM_Layer']
            after_tom = after['ToM_Layer']

            # Team Focus changes
            if 'Team_Focus' in before_tom and 'Team_Focus' in after_tom:
                before_focus = before_tom['Team_Focus']['focus_distribution']
                after_focus = after_tom['Team_Focus']['focus_distribution']

                for focus_type in before_focus:
                    if before_focus[focus_type] != after_focus[focus_type]:
                        diff_summary += f"  Team_Focus.{focus_type}: {before_focus[focus_type]:.2f} → {after_focus[focus_type]:.2f}\n"

            # First-order ToM changes
            if 'Teammates' in before_tom and 'Teammates' in after_tom:
                for player in before_tom['Teammates']:
                    if player in after_tom['Teammates']:
                        before_player = before_tom['Teammates'][player]
                        after_player = after_tom['Teammates'][player]

                        for attribute in before_player:
                            if before_player[attribute] != after_player[attribute]:
                                diff_summary += f"  {player}.{attribute}: {before_player[attribute]:.2f} → {after_player[attribute]:.2f}\n"

            # Second-order ToM changes (Variation 4: What others think about ME)
            if 'Recursive_ToM' in before_tom and 'Recursive_ToM' in after_tom:
                for player_key in before_tom['Recursive_ToM']:
                    if player_key in after_tom['Recursive_ToM']:
                        before_recursive = before_tom['Recursive_ToM'][player_key]
                        after_recursive = after_tom['Recursive_ToM'][player_key]

                        for me_player in before_recursive:
                            if me_player in after_recursive:
                                before_beliefs = before_recursive[me_player]
                                after_beliefs = after_recursive[me_player]

                                # Track skill assessment changes
                                if 'believes_about_my_skill' in before_beliefs and 'believes_about_my_skill' in after_beliefs:
                                    if before_beliefs['believes_about_my_skill'] != after_beliefs['believes_about_my_skill']:
                                        diff_summary += f"  {player_key}→{me_player}.skill_assessment: {before_beliefs['believes_about_my_skill']:.2f} → {after_beliefs['believes_about_my_skill']:.2f}\n"

                                # Track strategy assessment changes
                                if 'believes_about_my_strategy' in before_beliefs and 'believes_about_my_strategy' in after_beliefs:
                                    if before_beliefs['believes_about_my_strategy'] != after_beliefs['believes_about_my_strategy']:
                                        diff_summary += f"  {player_key}→{me_player}.strategy_assessment: {before_beliefs['believes_about_my_strategy']:.2f} → {after_beliefs['believes_about_my_strategy']:.2f}\n"

        return diff_summary