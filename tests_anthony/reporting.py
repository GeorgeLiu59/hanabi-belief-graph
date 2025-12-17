"""Reporting and visualization for ablation study results.

This module generates comprehensive markdown reports comparing agent performance
across different context representations.
"""

from typing import List, Dict, Any
from collections import defaultdict


def generate_ablation_report(results: List[Any]) -> str:
    """
    Generate comprehensive markdown report from ablation study results.

    Args:
        results: List of AgentDecisionMetrics objects

    Returns:
        Markdown-formatted report string
    """
    if not results:
        return "# Ablation Study Report\n\nNo results available yet. Run tests first.\n"

    # Group results by scenario and context
    by_scenario = defaultdict(list)
    by_context = defaultdict(list)

    for result in results:
        by_scenario[result.scenario_name].append(result)
        by_context[result.context_type].append(result)

    # Generate report sections
    report = _generate_header()
    report += _generate_executive_summary(results, by_context)
    report += _generate_scenario_analysis(by_scenario)
    report += _generate_context_analysis(by_context)
    report += _generate_turn_28_deep_dive(by_scenario.get("turn_28_white_stack_bug", []))
    report += _generate_recommendations(by_context)

    return report


def _generate_header() -> str:
    """Generate report header."""
    return """# Hanabi Belief Graph Ablation Study - Results

**Research Question:** Do belief graphs actually improve LLM agent decision-making in Hanabi,
or could simpler context representations work better?

**Methodology:** Compare agent performance with 5 different context representations on identical game scenarios.

---

"""


def _generate_executive_summary(results: List[Any], by_context: Dict[str, List[Any]]) -> str:
    """Generate executive summary with key findings."""
    summary = "## Executive Summary\n\n"

    # Calculate overall statistics
    total_tests = len(results)
    context_types = list(by_context.keys())

    summary += f"**Total Tests Run:** {total_tests} ({len(by_context)} context types × {total_tests // len(by_context)} scenarios)\n\n"

    # Rank contexts by safety
    context_safety_scores = {}
    for context_type, ctx_results in by_context.items():
        safe_count = sum(1 for r in ctx_results if r.action_safety == "SAFE")
        fatal_count = sum(1 for r in ctx_results if r.action_safety == "FATAL")
        total = len(ctx_results)
        safety_pct = (safe_count / total * 100) if total > 0 else 0
        fatal_pct = (fatal_count / total * 100) if total > 0 else 0

        context_safety_scores[context_type] = {
            "safety_pct": safety_pct,
            "fatal_pct": fatal_pct,
            "safe_count": safe_count,
            "fatal_count": fatal_count,
            "total": total
        }

    # Sort by safety percentage
    ranked_contexts = sorted(
        context_safety_scores.items(),
        key=lambda x: x[1]["safety_pct"],
        reverse=True
    )

    summary += "### Context Performance Ranking (by safety)\n\n"
    for rank, (context_type, scores) in enumerate(ranked_contexts, 1):
        emoji = "✅" if rank == 1 else "⚠️" if rank <= 3 else "❌"
        summary += f"{rank}. {emoji} **{_format_context_name(context_type)}**: "
        summary += f"{scores['safe_count']}/{scores['total']} safe ({scores['safety_pct']:.1f}%), "
        summary += f"{scores['fatal_count']} fatal ({scores['fatal_pct']:.1f}%)\n"

    summary += "\n"

    # Key findings
    best_context = ranked_contexts[0][0]
    worst_context = ranked_contexts[-1][0]

    summary += "### Key Findings\n\n"
    summary += f"- **Best performing context:** {_format_context_name(best_context)} "
    summary += f"({context_safety_scores[best_context]['safety_pct']:.1f}% safe decisions)\n"
    summary += f"- **Worst performing context:** {_format_context_name(worst_context)} "
    summary += f"({context_safety_scores[worst_context]['fatal_pct']:.1f}% fatal errors)\n"

    # Belief graph effectiveness
    if "full_belief_graph" in by_context and "raw_game_state" in by_context:
        belief_safe = context_safety_scores["full_belief_graph"]["safety_pct"]
        raw_safe = context_safety_scores["raw_game_state"]["safety_pct"]

        if belief_safe > raw_safe:
            summary += f"- **Belief graphs HELP**: Full belief graph ({belief_safe:.1f}% safe) outperforms raw game state ({raw_safe:.1f}% safe)\n"
        else:
            summary += f"- **Belief graphs HURT**: Raw game state ({raw_safe:.1f}% safe) outperforms full belief graph ({belief_safe:.1f}% safe)\n"

    summary += "\n---\n\n"
    return summary


def _generate_scenario_analysis(by_scenario: Dict[str, List[Any]]) -> str:
    """Generate per-scenario analysis comparing all contexts."""
    section = "## Scenario Analysis\n\n"
    section += "How did each context type perform on specific game scenarios?\n\n"

    for scenario_name, scenario_results in sorted(by_scenario.items()):
        if not scenario_results:
            continue

        section += f"### {_format_scenario_name(scenario_name)}\n\n"

        # Get ground truth
        ground_truth = scenario_results[0].ground_truth_action
        section += f"**Ground Truth Best Action:** {ground_truth.get('type', 'UNKNOWN')} "
        if 'target' in ground_truth:
            section += f"(target {ground_truth['target']})"
        section += f"\n\n**Reasoning:** {ground_truth.get('reasoning', 'N/A')}\n\n"

        # Table of results
        section += "| Context Type | Action | Safety | Used Beliefs | Coherence |\n"
        section += "|--------------|--------|--------|--------------|----------|\n"

        for result in scenario_results:
            context_name = _format_context_name(result.context_type)
            action_str = f"{result.action_type}"
            if result.action_target >= 0:
                action_str += f" (card {result.action_target})"

            safety_emoji = "✅" if result.action_safety == "SAFE" else "⚠️" if result.action_safety == "RISKY" else "❌"
            safety_str = f"{result.action_safety} {safety_emoji}"

            belief_usage = "Yes" if result.referenced_probabilities else "N/A"
            coherence_str = f"{result.logical_coherence_score:.2f}"

            section += f"| {context_name} | {action_str} | {safety_str} | {belief_usage} | {coherence_str} |\n"

        section += "\n"

        # Analysis
        safe_contexts = [r.context_type for r in scenario_results if r.action_safety == "SAFE"]
        fatal_contexts = [r.context_type for r in scenario_results if r.action_safety == "FATAL"]

        if safe_contexts:
            section += f"**Safe decisions:** {', '.join(_format_context_name(c) for c in safe_contexts)}\n\n"
        if fatal_contexts:
            section += f"**Fatal errors:** {', '.join(_format_context_name(c) for c in fatal_contexts)}\n\n"

        section += "---\n\n"

    return section


def _generate_context_analysis(by_context: Dict[str, List[Any]]) -> str:
    """Generate per-context analysis showing aggregate performance."""
    section = "## Context Type Analysis\n\n"
    section += "Aggregate performance for each context type across all scenarios.\n\n"

    for context_type, ctx_results in sorted(by_context.items()):
        section += f"### {_format_context_name(context_type)}\n\n"

        total = len(ctx_results)
        safe_count = sum(1 for r in ctx_results if r.action_safety == "SAFE")
        risky_count = sum(1 for r in ctx_results if r.action_safety == "RISKY")
        fatal_count = sum(1 for r in ctx_results if r.action_safety == "FATAL")

        safe_pct = (safe_count / total * 100) if total > 0 else 0
        fatal_pct = (fatal_count / total * 100) if total > 0 else 0

        section += f"**Overall Performance:**\n"
        section += f"- Safe decisions: {safe_count}/{total} ({safe_pct:.1f}%)\n"
        section += f"- Risky decisions: {risky_count}/{total} ({risky_count / total * 100:.1f}%)\n"
        section += f"- Fatal errors: {fatal_count}/{total} ({fatal_pct:.1f}%)\n\n"

        # Belief usage (if applicable)
        if context_type in ["full_belief_graph", "natural_language_belief", "certainty_only"]:
            belief_ref_count = sum(1 for r in ctx_results if r.referenced_probabilities)
            impossible_mention_count = sum(1 for r in ctx_results if r.mentioned_impossible_cards)

            section += f"**Belief Graph Usage:**\n"
            section += f"- Referenced probabilities: {belief_ref_count}/{total} ({belief_ref_count / total * 100:.1f}%)\n"
            section += f"- Mentioned impossible cards: {impossible_mention_count}/{total} ({impossible_mention_count / total * 100:.1f}%)\n\n"

        # Reasoning quality
        avg_coherence = sum(r.logical_coherence_score for r in ctx_results) / total
        avg_length = sum(r.reasoning_length_words for r in ctx_results) / total

        section += f"**Reasoning Quality:**\n"
        section += f"- Average coherence score: {avg_coherence:.2f}/1.0\n"
        section += f"- Average reasoning length: {avg_length:.0f} words\n\n"

        section += "---\n\n"

    return section


def _generate_turn_28_deep_dive(turn_28_results: List[Any]) -> str:
    """Generate detailed analysis of Turn 28 bug scenario."""
    if not turn_28_results:
        return ""

    section = "## Turn 28 Bug - Deep Dive Analysis\n\n"
    section += "**Critical Test:** Does the Turn 28 bug (uniform 20% probabilities for impossible cards) "
    section += "make belief graphs counterproductive?\n\n"

    section += "**Expected Bug Behavior:** Agent with full belief graph sees White-1 with 20% probability "
    section += "despite White stack at 3, leading to fatal PLAY action.\n\n"

    # Analyze which contexts fell victim to the bug
    bug_victims = []
    bug_avoiders = []

    for result in turn_28_results:
        is_fatal = (result.action_type == "PLAY" and result.action_target == 0)
        if is_fatal:
            bug_victims.append(result)
        else:
            bug_avoiders.append(result)

    section += "### Results:\n\n"
    section += f"**Contexts that FELL VICTIM to bug (played White-1):** {len(bug_victims)}/{len(turn_28_results)}\n\n"

    if bug_victims:
        for result in bug_victims:
            section += f"- ❌ **{_format_context_name(result.context_type)}**: Played card 0 (White-1) - FATAL ERROR\n"
            if result.referenced_probabilities:
                section += f"  - Mentioned probabilities in reasoning: Yes\n"
            # Show snippet of reasoning
            reasoning_snippet = result.reasoning_text[:200] + "..." if len(result.reasoning_text) > 200 else result.reasoning_text
            section += f"  - Reasoning excerpt: \"{reasoning_snippet}\"\n"

    section += f"\n**Contexts that AVOIDED the bug:** {len(bug_avoiders)}/{len(turn_28_results)}\n\n"

    if bug_avoiders:
        for result in bug_avoiders:
            section += f"- ✅ **{_format_context_name(result.context_type)}**: {result.action_type} card {result.action_target} - SAFE\n"

    # Conclusion
    section += "\n### Conclusion:\n\n"

    if len(bug_victims) > len(bug_avoiders):
        section += "**CRITICAL FINDING:** More contexts fell victim to the bug than avoided it. "
        section += "This suggests the Turn 28 bug significantly undermines belief graph effectiveness.\n\n"
    elif "full_belief_graph" in [r.context_type for r in bug_victims]:
        section += "**CRITICAL FINDING:** The full belief graph context led to a fatal error, "
        section += "while simpler contexts avoided the bug. This demonstrates that incorrect belief "
        section += "probabilities actively harm decision-making.\n\n"
    else:
        section += "Agents successfully avoided the Turn 28 bug across most contexts.\n\n"

    section += "---\n\n"
    return section


def _generate_recommendations(by_context: Dict[str, List[Any]]) -> str:
    """Generate actionable recommendations based on findings."""
    section = "## Recommendations\n\n"

    # Calculate context rankings
    context_safety = {}
    for context_type, ctx_results in by_context.items():
        total = len(ctx_results)
        safe_count = sum(1 for r in ctx_results if r.action_safety == "SAFE")
        safety_pct = (safe_count / total * 100) if total > 0 else 0
        context_safety[context_type] = safety_pct

    best_context = max(context_safety.items(), key=lambda x: x[1])[0]

    # Recommendation 1: Best context to use
    section += f"### 1. Use {_format_context_name(best_context)} Context\n\n"
    section += f"This context achieved the highest safety rate ({context_safety[best_context]:.1f}%) "
    section += f"across all tested scenarios. Consider prioritizing this representation format.\n\n"

    # Recommendation 2: Fix Turn 28 bug if belief graphs are best
    if best_context in ["full_belief_graph", "natural_language_belief"]:
        section += "### 2. Fix Turn 28 Bug IMMEDIATELY\n\n"
        section += "Belief-based contexts performed best, but the Turn 28 bug undermines their effectiveness. "
        section += "Priority should be fixing the `_create_belief_entry()` method at "
        section += "`belief_graph_base.py:862` to filter impossible cards.\n\n"
    elif "full_belief_graph" in context_safety:
        # Belief graphs performed poorly
        belief_safety = context_safety["full_belief_graph"]
        section += "### 2. Consider Alternative Representations\n\n"
        section += f"Full belief graph achieved only {belief_safety:.1f}% safety rate. "
        section += f"Consider switching to {_format_context_name(best_context)} representation "
        section += "which performed significantly better.\n\n"

    # Recommendation 3: Format considerations
    if "full_belief_graph" in context_safety and "natural_language_belief" in context_safety:
        json_safety = context_safety["full_belief_graph"]
        nl_safety = context_safety["natural_language_belief"]

        if abs(json_safety - nl_safety) > 15:  # Significant difference
            better_format = "natural language" if nl_safety > json_safety else "JSON"
            section += f"### 3. Optimize Belief Representation Format\n\n"
            section += f"There's a {abs(json_safety - nl_safety):.1f}% performance difference between "
            section += f"JSON and natural language formats. **{better_format.capitalize()} format performs better** "
            section += "for LLM reasoning. Consider reformatting belief graphs accordingly.\n\n"

    # Recommendation 4: Reasoning improvements
    section += "### 4. Improve Chain of Thought Prompting\n\n"
    section += "Analysis of reasoning quality across contexts can inform prompt engineering:\n"
    section += "- Encourage explicit reasoning about impossible cards\n"
    section += "- Prompt agents to cross-reference fireworks state with belief probabilities\n"
    section += "- Request step-by-step logical flow (premises → inference → conclusion)\n\n"

    section += "---\n\n"
    return section


# Helper formatting functions

def _format_context_name(context_type: str) -> str:
    """Convert context_type key to human-readable name."""
    names = {
        "full_belief_graph": "Full Belief Graph (JSON)",
        "certainty_only": "Certainty Only",
        "raw_game_state": "Raw Game State",
        "minimal_context": "Minimal Context",
        "natural_language_belief": "Natural Language Belief"
    }
    return names.get(context_type, context_type)


def _format_scenario_name(scenario_name: str) -> str:
    """Convert scenario_name key to human-readable name."""
    names = {
        "turn_1_game_start": "Turn 1: Game Start",
        "turn_5_minimal_fireworks": "Turn 5: Minimal Fireworks",
        "turn_10_medium_complexity": "Turn 10: Medium Complexity",
        "turn_28_white_stack_bug": "Turn 28: White Stack Bug (CRITICAL)",
        "turn_42_late_game": "Turn 42: Late Game",
        "info_tokens_depleted": "Info Tokens Depleted",
        "last_life_token": "Last Life Token"
    }
    return names.get(scenario_name, scenario_name)


def export_results_csv(results: List[Any], filepath: str):
    """Export results as CSV for further analysis."""
    import csv

    if not results:
        return

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'scenario_name', 'context_type', 'turn', 'action_type', 'action_target',
            'action_safety', 'matches_ground_truth', 'referenced_probabilities',
            'mentioned_impossible_cards', 'logical_coherence_score',
            'reasoning_length_words'
        ])

        writer.writeheader()

        for result in results:
            writer.writerow({
                'scenario_name': result.scenario_name,
                'context_type': result.context_type,
                'turn': result.turn,
                'action_type': result.action_type,
                'action_target': result.action_target,
                'action_safety': result.action_safety,
                'matches_ground_truth': result.matches_ground_truth,
                'referenced_probabilities': result.referenced_probabilities,
                'mentioned_impossible_cards': result.mentioned_impossible_cards,
                'logical_coherence_score': result.logical_coherence_score,
                'reasoning_length_words': result.reasoning_length_words
            })
