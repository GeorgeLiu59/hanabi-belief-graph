# Parallel Games Runner - Usage Guide

## Overview

The `async_game_runner.py` enables running multiple Hanabi games concurrently with comprehensive logging and experiment tracking.

## Features

- **Batched Parallel Execution**: Run up to 10 games concurrently (respects Gemini API limits)
- **Experiment Tagging**: Tag runs for easy identification and comparison
- **Game ID Tracking**: Each game gets a unique ID for log correlation
- **Comprehensive Logging**: Agent logs, game state, and events tracked separately
- **Crash Protection**: Automatic fallback to random moves if agents fail
- **Buffer Flushing**: All logs immediately written to disk for data integrity

## Usage

### Basic Command

```bash
python3 hanabi_learning_environment/async_game_runner.py <num_games> <players> <episodes_per_game> <agent_config> [max_turns] [--tag TAG]
```

### Parameters

- `num_games`: Total number of games to run (executed in batches of 10)
- `players`: Number of players per game (typically 2)
- `episodes_per_game`: Episodes per game (typically 1)
- `agent_config`: Comma-separated agent types (e.g., `'GeminiAgent,GeminiAgent'`)
- `max_turns`: (Optional) Maximum turns per game
- `--tag`: (Optional) Tag for this experiment run (spaces converted to underscores)

### Examples

#### 20 Games - Gemini Baseline
```bash
python3 hanabi_learning_environment/async_game_runner.py 20 2 1 'GeminiAgent,GeminiAgent' --tag "baseline_20games"
```

#### 10 Games - BeliefGraph Probabilistic vs Gemini
```bash
python3 hanabi_learning_environment/async_game_runner.py 10 2 1 'BeliefGraphAgent:probabilistic,GeminiAgent' --tag "bg_prob_vs_gemini"
```

#### With Turn Limit
```bash
python3 hanabi_learning_environment/async_game_runner.py 5 2 1 'GeminiAgent,GeminiAgent' 50 --tag "test_run"
```

## Log Structure

All logs use the new structured format with game IDs and tags for parallel execution tracking.

### Agent Logs

**Location**: `logs/{model_type}_{mode}/`

**Filename Format**: `DATE_TIME_LLMMODEL_MODEL_MODE_TAG_GAMEID_AGENTID.log`

**Example**:
```
logs/gemini_base/20260103_120806_2_5_gemini_pro_Gemini_base_baseline_20games_game0_3161.log
logs/gemini_base/20260103_120806_2_5_gemini_pro_Gemini_base_baseline_20games_game0_6195.log
```

**Subdirectories**:
- `gemini_base/` - GeminiAgent logs
- `bg_probabilistic/` - BeliefGraph probabilistic variant logs
- `bg_certainty/` - BeliefGraph certainty variant logs
- `bg_theory_of_mind/` - BeliefGraph ToM variant logs

### Game State Logs

**Location**: `logs/game_state/DATE_TIME_AGENT1-AGENT2/`

**Filename Format**: `game_state_DATE_TIME_gameID.jsonl`

**Example**:
```
logs/game_state/20260103_120806_Gemini_base-Gemini_base/game_state_20260103_120806_game0.jsonl
logs/game_state/20260103_120806_Gemini_base-Gemini_base/game_state_20260103_120806_game1.jsonl
```

**Content**: Turn-by-turn game state (fireworks, lives, clues, deck size)

### Event Logs

**Location**: `logs/events/DATE_TIME_AGENT1-AGENT2/`

**Filename Format**: `events_DATE_TIME_gameID.jsonl`

**Example**:
```
logs/events/20260103_120806_Gemini_base-Gemini_base/events_20260103_120806_game0.jsonl
logs/events/20260103_120806_Gemini_base-Gemini_base/events_20260103_120806_game1.jsonl
```

**Content**: Life losses, clue usage, firework progress with timestamps

## Identifying Logs from Same Game

All logs from the same game share the same `gameID`:

**Game 0 Logs**:
- Agent 1: `..._game0_3161.log`
- Agent 2: `..._game0_6195.log`
- Game State: `game_state_..._game0.jsonl`
- Events: `events_..._game0.jsonl`

**Game 1 Logs**:
- Agent 1: `..._game1_1000.log`
- Agent 2: `..._game1_4731.log`
- Game State: `game_state_..._game1.jsonl`
- Events: `events_..._game1.jsonl`

## Batched Execution

Games run in batches of 10 to respect Gemini API concurrent request limits:

**Example: 20 Games**
```
Batch 1: Games 0-9  (runs in parallel)
   ↓
Batch 2: Games 10-19 (runs in parallel)
   ↓
All 20 games complete!
```

**Runtime**: ~1-1.5 hours per batch (depends on game length and API response times)

## Error Handling

The system has three layers of crash protection:

### Layer 1: Agent Retry Logic
- **5 retries** with exponential backoff (1s → 2s → 4s → 8s → 16s)
- Handles API timeouts, parsing errors, illegal moves

### Layer 2: Game-Level Fallback
- If agent fails after all retries, **random legal move** selected
- Game continues instead of crashing
- Error logged for analysis

### Layer 3: Coordinator Exception Handling
- If entire game crashes, other games continue
- Full traceback logged
- Summary shows failed games count

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch agent logs live
tail -f logs/gemini_base/*.log

# Check for errors
grep -i "error\|failed\|504" logs/gemini_base/*.log
```

### Post-Run Analysis
```bash
# Count completed games
ls logs/gemini_base/*_baseline_20games_*.log | wc -l

# Check for API errors
grep "504 Deadline\|ALL_RETRIES_FAILED" logs/gemini_base/*.log
```

## Visualization

Use the `visualize_lives.py` script to plot life token losses:

```bash
# Process all logs in a subdirectory
python scripts/visualize_lives.py logs/gemini_base --out output_directory/

# Process specific game
python scripts/visualize_lives.py logs/gemini_base/*_game0_*.log --out game0_lives.png
```

## API Limits

### Gemini API Constraints

**Free Tier**:
- 2 requests/minute - ❌ Not suitable for parallel games
- 1,500 requests/day

**Paid Tier**:
- 1,000 requests/minute - ✅ Handles 10+ concurrent games
- 4,000 requests/day limit

**Concurrent Requests**:
- ~10-13 concurrent connections max
- System automatically batches to stay under this limit

### Expected API Usage

For 20 games (2 players each):
- **Total agents**: 40
- **Concurrent agents**: ~10-20 (only current player acts)
- **Requests per turn**: 1-6 (with retries)
- **Total requests**: ~2,000-5,000 (depends on game length)

## Troubleshooting

### Issue: Games stop at ~13 concurrent
**Cause**: Gemini API concurrent connection limit
**Solution**: System now batches automatically (no action needed)

### Issue: "504 Deadline Exceeded" errors
**Cause**: Gemini API timeout (not rate limit)
**Solution**: Retry logic handles this automatically

### Issue: Agent logs missing
**Cause**: Check if agent crashed before initializing logger
**Solution**: Check coordinator error output for exceptions

### Issue: Can't identify which logs belong to same game
**Cause**: Old logs without game_id
**Solution**: New logs include `gameID` in filename

## Best Practices

1. **Start small**: Test with 5 games before scaling to 20
2. **Use tags**: Always tag experiments for easy comparison
3. **Monitor errors**: Check for API timeouts and failures
4. **Check logs exist**: Verify all games produced logs before analysis
5. **Clean old logs**: Archive or delete old test runs to avoid confusion

## File Name Components

**Format**: `DATE_TIME_LLMMODEL_MODEL_MODE_TAG_GAMEID_AGENTID.log`

- `DATE_TIME`: Session start time (YYYYMMDD_HHMMSS)
- `LLMMODEL`: LLM version (e.g., "2_5_gemini_pro")
- `MODEL`: Model type ("BG" or "Gemini")
- `MODE`: Variant ("base", "probabilistic", "certainty", "theory_of_mind")
- `TAG`: Your experiment tag (optional, spaces → underscores)
- `GAMEID`: Game number (game0, game1, ...)
- `AGENTID`: Unique agent identifier (4-digit random number)

**Example Breakdown**:
```
20260103_120806_2_5_gemini_pro_Gemini_base_baseline_20games_game5_3516.log
│           │        │            │      │      │               │     │
│           │        │            │      │      │               │     └─ Agent ID: 3516
│           │        │            │      │      │               └─────── Game 5
│           │        │            │      │      └───────────────────── Tag: baseline_20games
│           │        │            │      └────────────────────────── Mode: base
│           │        │            └───────────────────────────────── Model: Gemini
│           │        └────────────────────────────────────────────── LLM: 2.5 Gemini Pro
│           └─────────────────────────────────────────────────────── Time: 12:08:06
└─────────────────────────────────────────────────────────────────── Date: 2026-01-03
```
