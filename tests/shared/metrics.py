"""Metrics capture system for natural language belief ablation study.

This module defines data structures for capturing agent decision metrics
and aggregating results across game stages (early, mid, late).

Modified for stage-based analysis (removed multi-context comparison).
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
from pathlib import Path


@dataclass
class AgentDecisionMetrics:
    """Captures all relevant metrics for a single agent decision."""

    # Scenario identification
    scenario_name: str
    game_stage: str  # "early_game", "mid_game", "late_game"
    turn: int

    # Decision outputs
    action_type: str  # "PLAY", "DISCARD", "REVEAL_COLOR", "REVEAL_RANK"
    action_target: int  # Card index or player offset
    action_details: Dict[str, Any]  # Full action object
    reasoning_text: str

    # Safety evaluation
    action_safety: str  # "SAFE", "RISKY", "FATAL"
    would_lose_life: bool
    is_optimal: bool
    matches_ground_truth: bool

    # LLM-as-Judge evaluation (optional - only when judged)
    judge_score: Optional[float]  # Composite score 0.0-1.0
    judge_evaluation: Optional[Dict[str, Any]]  # Full evaluation dict with FA, IR, LS, AA

    # Reasoning quality (heuristic)
    reasoning_length_words: int
    references_fireworks: bool
    references_discards: bool
    references_teammate_hand: bool
    logical_coherence_score: float  # 0.0 to 1.0 (heuristic)

    # Ground truth comparison
    ground_truth_action: Dict[str, Any]
    dangerous_actions: List[Dict[str, Any]]

    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return asdict(self)

    @classmethod
    def from_agent_output(
        cls,
        scenario: Dict[str, Any],
        game_stage: str,
        action: Dict[str, Any],
        reasoning: str,
        safety: str,
        reasoning_quality: Dict[str, Any],
        judge_evaluation: Optional[Dict[str, Any]] = None
    ) -> "AgentDecisionMetrics":
        """
        Factory method to create metrics from agent outputs.

        Args:
            scenario: Game scenario from scenarios.json
            game_stage: "early_game", "mid_game", or "late_game"
            action: Agent's chosen action
            reasoning: Agent's reasoning text
            safety: "SAFE", "RISKY", or "FATAL"
            reasoning_quality: Dict with reasoning quality metrics
            judge_evaluation: Optional dict from LLM-as-Judge (if judged)

        Returns:
            AgentDecisionMetrics instance with all fields populated
        """
        ground_truth = scenario["ground_truth_best_action"]

        return cls(
            # Scenario identification
            scenario_name=scenario["scenario_name"],
            game_stage=game_stage,
            turn=scenario["turn"],
            # Decision outputs
            action_type=action.get("type", "UNKNOWN"),
            action_target=action.get("target", -1),
            action_details=action,
            reasoning_text=reasoning,
            # Safety evaluation
            action_safety=safety,
            would_lose_life=(safety == "FATAL"),
            is_optimal=(safety == "SAFE" and action.get("type") == ground_truth.get("type")),
            matches_ground_truth=(
                action.get("type") == ground_truth.get("type")
                and action.get("target") == ground_truth.get("target")
            ),
            # Judge evaluation
            judge_score=judge_evaluation.get("composite_score") if judge_evaluation else None,
            judge_evaluation=judge_evaluation,
            # Reasoning quality
            reasoning_length_words=reasoning_quality.get("length_words", 0),
            references_fireworks=reasoning_quality.get("mentions_fireworks", False),
            references_discards=reasoning_quality.get("mentions_discards", False),
            references_teammate_hand=reasoning_quality.get("mentions_teammate", False),
            logical_coherence_score=reasoning_quality.get("coherence_score", 0.0),
            # Ground truth
            ground_truth_action=ground_truth,
            dangerous_actions=scenario.get("dangerous_actions", [])
        )


class MetricsCollector:
    """Collects and aggregates metrics across all tests."""

    def __init__(self):
        self.results: List[AgentDecisionMetrics] = []

    def add_result(self, metrics: AgentDecisionMetrics):
        """Add a single test result."""
        self.results.append(metrics)

    def get_results_by_scenario(self, scenario_name: str) -> List[AgentDecisionMetrics]:
        """Get all results for a specific scenario."""
        return [r for r in self.results if r.scenario_name == scenario_name]

    def get_results_by_stage(self, game_stage: str) -> List[AgentDecisionMetrics]:
        """Get all results for a specific game stage."""
        return [r for r in self.results if r.game_stage == game_stage]

    def get_unique_scenarios(self) -> List[str]:
        """Get list of unique scenario names tested."""
        return list(set(r.scenario_name for r in self.results))

    def get_unique_stages(self) -> List[str]:
        """Get list of unique game stages tested."""
        return list(set(r.game_stage for r in self.results))

    def calculate_stage_performance(self, game_stage: str) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics for a game stage.

        Args:
            game_stage: "early_game", "mid_game", or "late_game"

        Returns:
            Dict with performance statistics
        """
        results = self.get_results_by_stage(game_stage)

        if not results:
            return {"error": "No results for this game stage"}

        total = len(results)
        safe_count = sum(1 for r in results if r.action_safety == "SAFE")
        risky_count = sum(1 for r in results if r.action_safety == "RISKY")
        fatal_count = sum(1 for r in results if r.action_safety == "FATAL")
        optimal_count = sum(1 for r in results if r.is_optimal)
        matches_gt_count = sum(1 for r in results if r.matches_ground_truth)

        # Judge evaluation statistics (only for judged results)
        judged_results = [r for r in results if r.judge_score is not None]
        judged_count = len(judged_results)
        avg_judge_score = (
            sum(r.judge_score for r in judged_results) / judged_count
            if judged_count > 0 else None
        )

        # Reasoning quality
        avg_coherence = sum(r.logical_coherence_score for r in results) / total
        avg_length = sum(r.reasoning_length_words for r in results) / total

        return {
            "total_decisions": total,
            "safety_breakdown": {
                "safe": safe_count,
                "risky": risky_count,
                "fatal": fatal_count,
                "safe_percentage": (safe_count / total * 100) if total > 0 else 0,
                "fatal_percentage": (fatal_count / total * 100) if total > 0 else 0
            },
            "accuracy": {
                "optimal_count": optimal_count,
                "matches_ground_truth": matches_gt_count,
                "optimal_percentage": (optimal_count / total * 100) if total > 0 else 0
            },
            "judge_evaluation": {
                "judged_count": judged_count,
                "judge_percentage": (judged_count / total * 100) if total > 0 else 0,
                "avg_composite_score": avg_judge_score,
            },
            "reasoning_quality": {
                "avg_coherence_score": avg_coherence,
                "avg_length_words": avg_length
            }
        }

    def compare_stages_on_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Get performance on a specific scenario (typically only one stage per scenario).

        Args:
            scenario_name: Name of scenario to analyze

        Returns:
            Dict with scenario performance details
        """
        scenario_results = self.get_results_by_scenario(scenario_name)

        if not scenario_results:
            return {"error": "No results for this scenario"}

        # Typically one result per scenario in stage-based testing
        result = scenario_results[0]
        return {
            "scenario_name": scenario_name,
            "game_stage": result.game_stage,
            "action": f"{result.action_type} target={result.action_target}",
            "safety": result.action_safety,
            "matches_ground_truth": result.matches_ground_truth,
            "judge_score": result.judge_score,
            "coherence_score": result.logical_coherence_score
        }

    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        Generate overall summary statistics across all tests.

        Returns:
            Dict with aggregate statistics
        """
        if not self.results:
            return {"error": "No results collected yet"}

        stages = self.get_unique_stages()
        scenarios = self.get_unique_scenarios()

        # Calculate per-stage performance
        stage_performance = {
            stage: self.calculate_stage_performance(stage)
            for stage in stages
        }

        # Rank stages by safety
        stage_rankings = sorted(
            stages,
            key=lambda stage: stage_performance[stage]["safety_breakdown"]["safe_percentage"],
            reverse=True
        )

        return {
            "total_tests_run": len(self.results),
            "unique_scenarios": len(scenarios),
            "unique_stages": len(stages),
            "stage_performance": stage_performance,
            "stage_rankings_by_safety": stage_rankings,
            "scenarios_tested": scenarios
        }

    def export_to_json(self, filepath: str):
        """Export all results to JSON file for further analysis."""
        data = {
            "summary": self.generate_summary_stats(),
            "detailed_results": [r.to_dict() for r in self.results]
        }

        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_stage_comparison_report(self) -> str:
        """
        Generate markdown comparison report across game stages.

        Returns:
            Markdown-formatted report string
        """
        stats = self.generate_summary_stats()

        if "error" in stats:
            return f"# Error\n\n{stats['error']}"

        report_lines = []
        report_lines.append("# Natural Language Belief Ablation Study Results\n")
        report_lines.append(f"Total tests run: {stats['total_tests_run']}\n")
        report_lines.append(f"Unique scenarios: {stats['unique_scenarios']}\n")
        report_lines.append(f"Game stages tested: {', '.join(stats['unique_stages'])}\n")
        report_lines.append("\n## Performance by Game Stage\n")

        for stage in ["early_game", "mid_game", "late_game"]:
            if stage not in stats["stage_performance"]:
                continue

            perf = stats["stage_performance"][stage]
            report_lines.append(f"\n### {stage.replace('_', ' ').title()}\n")
            report_lines.append(f"- Total decisions: {perf['total_decisions']}")
            report_lines.append(f"- Safe actions: {perf['safety_breakdown']['safe']} ({perf['safety_breakdown']['safe_percentage']:.1f}%)")
            report_lines.append(f"- Fatal actions: {perf['safety_breakdown']['fatal']} ({perf['safety_breakdown']['fatal_percentage']:.1f}%)")
            report_lines.append(f"- Matches ground truth: {perf['accuracy']['matches_ground_truth']} ({perf['accuracy']['optimal_percentage']:.1f}%)")

            if perf['judge_evaluation']['judged_count'] > 0:
                report_lines.append(f"- Judged by LLM: {perf['judge_evaluation']['judged_count']} ({perf['judge_evaluation']['judge_percentage']:.1f}%)")
                if perf['judge_evaluation']['avg_composite_score'] is not None:
                    report_lines.append(f"- Avg judge score: {perf['judge_evaluation']['avg_composite_score']:.2f}/1.0")

            report_lines.append(f"- Avg coherence: {perf['reasoning_quality']['avg_coherence_score']:.2f}/1.0")
            report_lines.append(f"- Avg reasoning length: {perf['reasoning_quality']['avg_length_words']:.0f} words\n")

        return "\n".join(report_lines)
