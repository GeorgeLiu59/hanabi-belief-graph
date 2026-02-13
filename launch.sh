#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Interactive Hanabi Agent Setup ==="
echo ""

# Get number of players
while true; do
    read -p "Number of players (2-5): " PLAYERS
    if [[ "$PLAYERS" =~ ^[2-5]$ ]]; then
        break
    else
        echo "Please enter a number between 2 and 5"
    fi
done

# Get number of episodes
while true; do
    read -p "Number of episodes to play: " EPISODES
    if [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
        break
    else
        echo "Please enter a positive number"
    fi
done

echo ""
echo "Available agent types:"
echo "  1) GeminiAgent (requires API key)"
echo "  2) BeliefGraph-Certainty (Variation 1)"
echo "  3) BeliefGraph-Probabilistic (Variation 2)" 
echo "  4) BeliefGraph-Theory_of_Mind (Variation 3)"
echo "  5) SimpleAgent (rule-based)"
echo "  6) RandomAgent (random moves)"
echo ""

# Get agent configuration for each player
AGENT_CONFIG=""
for i in $(seq 1 $PLAYERS); do
    while true; do
        read -p "Player $i agent type (1-6): " choice
        case $choice in
            1) AGENT_CONFIG="$AGENT_CONFIG,GeminiAgent"; break ;;
            2) AGENT_CONFIG="$AGENT_CONFIG,BeliefGraphAgent:certainty"; break ;;
            3) AGENT_CONFIG="$AGENT_CONFIG,BeliefGraphAgent:probabilistic"; break ;;
            4) AGENT_CONFIG="$AGENT_CONFIG,BeliefGraphAgent:theory_of_mind"; break ;;
            5) AGENT_CONFIG="$AGENT_CONFIG,SimpleAgent"; break ;;
            6) AGENT_CONFIG="$AGENT_CONFIG,RandomAgent"; break ;;
            *) echo "Please enter 1-6";;
        esac
    done
done

# Remove leading comma
AGENT_CONFIG=${AGENT_CONFIG:1}

echo ""
echo "Configuration:"
echo "  Players: $PLAYERS"
echo "  Episodes: $EPISODES"
echo "  Agents: $AGENT_CONFIG"
echo ""
read -p "Start game? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled"
    exit 0
fi

LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

GAME_LOG="$LOG_DIR/game_${TIMESTAMP}.log"
SETUP_LOG="$LOG_DIR/setup_${TIMESTAMP}.log"
ERROR_LOG="$LOG_DIR/errors_${TIMESTAMP}.log"
VENV_DIR="venv"

echo "Cleaning previous processes..."
pkill -9 -f "rl_env_example.py" 2>/dev/null || true
pkill -9 -f "python.*hanabi" 2>/dev/null || true
pkill -9 -f "GeminiAgent" 2>/dev/null || true
pkill -9 -f "BeliefGraphAgent" 2>/dev/null || true
sleep 1

echo "Cleaning previous logs..."

# Force remove all files in logs directory
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR" -type f -delete 2>/dev/null || true
    rm -f "$LOG_DIR"/* 2>/dev/null || true
fi
mkdir -p "$LOG_DIR" 2>/dev/null

log_setup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SETUP: $1" >> "$SETUP_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$ERROR_LOG"
}


log_setup "Starting Interactive Hanabi Launch (Players: $PLAYERS, Episodes: $EPISODES, Config: $AGENT_CONFIG)"

if [ ! -d "$VENV_DIR" ]; then
    log_setup "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" 2>&1 | tee -a "$SETUP_LOG" || {
        log_error "Failed to create virtual environment"
        exit 1
    }
fi

log_setup "Activating virtual environment..."
source "$VENV_DIR/bin/activate" || {
    log_error "Failed to activate virtual environment"
    exit 1
}


log_setup "Upgrading pip..."
pip install --upgrade pip >> "$SETUP_LOG" 2>&1

log_setup "Installing build dependencies..."
pip install setuptools wheel scikit-build cmake ninja >> "$SETUP_LOG" 2>&1 || {
    log_error "Failed to install build dependencies"
    exit 1
}

log_setup "Cleaning previous builds..."
rm -rf _skbuild _cmake_test_compile build >> "$SETUP_LOG" 2>&1

log_setup "Installing hanabi_learning_environment package..."
pip install . >> "$SETUP_LOG" 2>&1 || {
    log_error "Failed to install hanabi_learning_environment package. Check $SETUP_LOG for details."
    exit 1
}

log_setup "Installing runtime dependencies..."
if [ "$NEEDS_GEMINI" = true ]; then
    pip install numpy google-generativeai python-dotenv >> "$SETUP_LOG" 2>&1 || {
        log_error "Failed to install Gemini dependencies"
        exit 1
    }
else
    pip install numpy >> "$SETUP_LOG" 2>&1 || {
        log_error "Failed to install basic dependencies"
        exit 1
    }
fi

log_setup "Dependencies installed successfully"

log_setup "Running mixed agents game (Players: $PLAYERS, Episodes: $EPISODES, Config: $AGENT_CONFIG)"

python hanabi_learning_environment/mixed_agents_runner.py "$PLAYERS" "$EPISODES" "$AGENT_CONFIG" \
    >> "$GAME_LOG" 2>&1 || {
    log_error "Mixed agents game execution failed"
    exit 1
}

log_setup "Game completed successfully"

# Extract all episode fireworks scores
ALL_EPISODE_SCORES=$(grep -o "Fireworks built this episode: [0-9]*" "$GAME_LOG" | grep -o "[0-9]*" || echo "0")
FINAL_SCORE=$(echo "$ALL_EPISODE_SCORES" | tail -1)
FINAL_SCORE_INT=$(echo "$FINAL_SCORE" | cut -d. -f1)

# Calculate average
if [ -n "$ALL_EPISODE_SCORES" ] && [ "$ALL_EPISODE_SCORES" != "0" ]; then
    TOTAL=$(echo "$ALL_EPISODE_SCORES" | awk '{sum += $1} END {print sum}')
    COUNT=$(echo "$ALL_EPISODE_SCORES" | wc -l)
    AVERAGE=$(echo "scale=1; $TOTAL / $COUNT" | bc -l 2>/dev/null || echo "0")
    AVERAGE_INT=$(echo "$AVERAGE" | cut -d. -f1)
else
    AVERAGE_INT="0"
fi 

echo "[Mixed Agents] Episodes: $(echo "$ALL_EPISODE_SCORES" | tr '\n' ' ' | sed 's/ $//' | sed 's/\.[0-9]*//g') | Avg Fireworks: $AVERAGE_INT out of 25" | tee -a "$GAME_LOG"