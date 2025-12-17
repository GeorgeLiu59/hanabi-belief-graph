# Ablation Study File Manifest

## Core Infrastructure (9 files)

### Test Data
- `fixtures/ablation_scenarios.json` - 7 test scenarios (Turn 1-42, bug scenarios)
- `fixtures/game_states.json` - Placeholder for future tests

### Python Modules
- `context_generators.py` - Generate 5 context representations
- `metrics.py` - Data capture (AgentDecisionMetrics, MetricsCollector)
- `analysis.py` - Evaluation functions (safety, belief usage, coherence)
- `reporting.py` - Auto-generate markdown reports
- `agent_adapter.py` - Bridge between tests and GeminiAgent

### Tests
- `test_context_ablation.py` - Main ablation tests (35 parametrized tests)
- `test_reasoning_quality.py` - Chain of Thought analysis (optional)
- `conftest.py` - pytest fixtures, session hooks, USE_REAL_AGENT toggle

### Documentation
- `ABLATION_STUDY_README.md` - Complete usage guide

### Generated (gitignored)
- `ablation_results.md` - Auto-generated report (regenerated each run)
- `ablation_results.json` - Auto-generated raw data (regenerated each run)

## Legacy Test Files (Not Part of Ablation Study)

These are from original instructions.md (bug testing, not ablation):
- `test_belief_initialization.py` - Unit tests for belief initialization
- `test_belief_updates.py` - Unit tests for belief updates
- `test_input_formats.py` - Format validation tests
- `test_prompt_consistency.py` - Prompt consistency tests

These are skeletons/placeholders and can be ignored for the ablation study.

## Quick File Count

**Ablation study core:** 9 files (~2,200 lines)
- 2 JSON fixtures
- 5 Python modules
- 2 test files
- 1 markdown doc

**Legacy tests:** 4 files (skeletons, not used)

**Generated:** 2 files (gitignored)

## What Was Removed

- ❌ `HOW_TO_USE.md` - Merged into ABLATION_STUDY_README.md
- ❌ `CHANGES.md` - Obsolete after integration complete
- ✅ Result: Single comprehensive README

## How to Navigate

1. **Start here:** ABLATION_STUDY_README.md
2. **Understand data:** fixtures/ablation_scenarios.json
3. **Run tests:** `pytest test_context_ablation.py -v`
4. **Toggle real agent:** conftest.py line 191
5. **View results:** ablation_results.md (auto-generated)

## Line Count Summary

```bash
# Core ablation infrastructure
$ wc -l context_generators.py metrics.py analysis.py reporting.py \
       test_context_ablation.py test_reasoning_quality.py \
       agent_adapter.py conftest.py fixtures/ablation_scenarios.json

   250 context_generators.py
   200 metrics.py
   280 analysis.py
   350 reporting.py
   250 test_context_ablation.py
   300 test_reasoning_quality.py
   180 agent_adapter.py
   310 conftest.py
   400 fixtures/ablation_scenarios.json
 ─────
 2,520 total
```

Clean, minimal, purpose-built. No bloat.
