"""Shared utilities for natural language belief graph ablation study.

This package contains common infrastructure used across early, mid, and late
game stage tests:

- game_stage_classifier.py: Hybrid game stage classification
- llm_judge.py: LLM-as-Judge evaluation protocol
- nl_belief_generator.py: Natural language belief description formatter
- agent_adapter.py: Bridge between test scenarios and agent interface
- metrics.py: Data collection and aggregation (adapted for stages)
- analysis.py: Evaluation functions (stage-based analysis)
- conftest.py: Shared pytest fixtures
"""

__version__ = "1.0.0"
__authors__ = ["Anthony", "Yash"]
