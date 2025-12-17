"""Metrics capture system for ablation study.

This module defines data structures for capturing agent decision metrics
and aggregating results across multiple test runs for comparison.
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
    context_type: str
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

    # Belief graph usage (only applicable for belief-based contexts)
    referenced_probabilities: bool
    mentioned_impossible_cards: bool
    used_numerical_probs: bool  # "20%" vs "unlikely"
    probability_interpretation_correct: bool

    # Reasoning quality
    reasoning_length_words: int
    references_fireworks: bool
    references_discards: bool
    references_teammate_hand: bool
    logical_coherence_score: float  # 0.0 to 1.0

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
        context_type: str,
        action: Dict[str, Any],
        reasoning: str,
        safety: str,
        belief_usage: Dict[str, Any],
        reasoning_quality: Dict[str, Any]
    ) -> "AgentDecisionMetrics":
        """
        Factory method to create metrics from agent outputs.

        Args:
            scenario: Game scenario from ablation_scenarios.json
            context_type: Type of context used
            action: Agent's chosen action
            reasoning: Agent's reasoning text
            safety: "SAFE", "RISKY", or "FATAL"
            belief_usage: Dict with belief usage analysis results
            reasoning_quality: Dict with reasoning quality metrics

        Returns:
            AgentDecisionMetrics instance with all fields populated
        """
        ground_truth = scenario["ground_truth_best_action"]

        return cls(
            # Scenario identification
            scenario_name=scenario["scenario_name"],
            context_type=context_type,
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
            # Belief graph usage
            referenced_probabilities=belief_usage.get("mentioned_probabilities", False),
            mentioned_impossible_cards=belief_usage.get("mentioned_impossible", False),
            used_numerical_probs=belief_usage.get("used_numerical_values", False),
            probability_interpretation_correct=belief_usage.get(
                "interpretation_correct", False
            ),
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

    def get_results_by_context(self, context_type: str) -> List[AgentDecisionMetrics]:
        """Get all results for a specific context type."""
        return [r for r in self.results if r.context_type == context_type]

    def get_unique_scenarios(self) -> List[str]:
        """Get list of unique scenario names tested."""
        return list(set(r.scenario_name for r in self.results))

    def get_unique_contexts(self) -> List[str]:
        """Get list of unique context types tested."""
        return list(set(r.context_type for r in self.results))

    def calculate_context_performance(self, context_type: str) -> Dict[str, Any]:
        """
        Calculate aggregate performance metrics for a context type.

        Returns:
            Dict with performance statistics
        """
        results = self.get_results_by_context(context_type)

        if not results:
            return {"error": "No results for this context type"}

        total = len(results)
        safe_count = sum(1 for r in results if r.action_safety == "SAFE")
        risky_count = sum(1 for r in results if r.action_safety == "RISKY")
        fatal_count = sum(1 for r in results if r.action_safety == "FATAL")
        optimal_count = sum(1 for r in results if r.is_optimal)
        matches_gt_count = sum(1 for r in results if r.matches_ground_truth)

        # Belief usage (only meaningful for belief-based contexts)
        belief_ref_count = sum(1 for r in results if r.referenced_probabilities)
        impossible_mention_count = sum(1 for r in results if r.mentioned_impossible_cards)

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
            "belief_usage": {
                "referenced_probabilities": belief_ref_count,
                "mentioned_impossible_cards": impossible_mention_count,
                "probability_usage_rate": (belief_ref_count / total * 100) if total > 0 else 0
            },
            "reasoning_quality": {
                "avg_coherence_score": avg_coherence,
                "avg_length_words": avg_length
            }
        }

    def compare_contexts_on_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """
        Compare how different contexts performed on the same scenario.

        Args:
            scenario_name: Name of scenario to analyze

        Returns:
            Dict mapping context_type to its performance on this scenario
        """
        scenario_results = self.get_results_by_scenario(scenario_name)

        comparison = {}
        for result in scenario_results:
            comparison[result.context_type] = {
                "action": f"{result.action_type} target={result.action_target}",
                "safety": result.action_safety,
                "matches_ground_truth": result.matches_ground_truth,
                "referenced_beliefs": result.referenced_probabilities,
                "coherence_score": result.logical_coherence_score
            }

        return comparison

    def generate_summary_stats(self) -> Dict[str, Any]:
        """
        Generate overall summary statistics across all tests.

        Returns:
            Dict with aggregate statistics
        """
        if not self.results:
            return {"error": "No results collected yet"}

        contexts = self.get_unique_contexts()
        scenarios = self.get_unique_scenarios()

        # Calculate per-context performance
        context_performance = {
            ctx: self.calculate_context_performance(ctx)
            for ctx in contexts
        }

        # Rank contexts by safety
        context_rankings = sorted(
            contexts,
            key=lambda ctx: context_performance[ctx]["safety_breakdown"]["safe_percentage"],
            reverse=True
        )

        return {
            "total_tests_run": len(self.results),
            "unique_scenarios": len(scenarios),
            "unique_contexts": len(contexts),
            "context_performance": context_performance,
            "context_rankings_by_safety": context_rankings,
            "scenarios_tested": scenarios
        }

    def export_to_json(self, filepath: str):
        """Export all results to JSON file for further analysis."""
        data = {
            "summary": self.generate_summary_stats(),
            "detailed_results": [r.to_dict() for r in self.results]
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def generate_comparison_report(self) -> str:
        """
        Generate comprehensive markdown comparison report.

        Returns:
            Markdown-formatted report string
        """
        # This will be implemented in reporting.py
        # Placeholder for now
        from .reporting import generate_ablation_report
        return generate_ablation_report(self.results)
