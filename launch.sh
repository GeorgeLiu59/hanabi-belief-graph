#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
sleep 1

echo "Cleaning previous logs..."

rm -rf "$LOG_DIR"/* 2>/dev/null
mkdir -p "$LOG_DIR" 2>/dev/null

log_setup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] SETUP: $1" >> "$SETUP_LOG"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$ERROR_LOG"
}

log_setup "Starting Hanabi Gemini Agent Launch"

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

if [ ! -f ".env" ]; then
    log_setup "No .env file found. Creating template..."
    cat > .env << 'EOF'
GEMINI_API_KEY=your_gemini_api_key_here
EOF
    log_error ".env file created. Please add your GEMINI_API_KEY and run again."
    exit 1
fi

if ! grep -q "GEMINI_API_KEY=" .env || grep -q "your_gemini_api_key_here" .env; then
    log_error "Please set your GEMINI_API_KEY in .env file"
    exit 1
fi

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
pip install numpy google-generativeai python-dotenv >> "$SETUP_LOG" 2>&1 || {
    log_error "Failed to install required dependencies"
    exit 1
}

log_setup "Dependencies installed successfully"

PLAYERS=${1:-2}

log_setup "Running Hanabi game with GeminiAgent (Players: $PLAYERS, Episodes: 3)"

python examples/rl_env_example.py \
    --agent_class=GeminiAgent \
    --players="$PLAYERS" \
    --num_episodes=3 \
    >> "$GAME_LOG" 2>&1 || {
    log_error "Game execution failed"
    exit 1
}

log_setup "Game completed successfully"

# Extract all episode scores
ALL_EPISODE_SCORES=$(grep -o "Max Reward: [0-9]*\.[0-9]*" "$GAME_LOG" | grep -o "[0-9]*\.[0-9]*" || echo "0")
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

# Log results to game log file only
{
    echo ""
    echo "================================"
    echo "HANABI GAME RESULTS"
    echo "================================"
    echo "Episode Scores:"
    EPISODE_NUM=0
    echo "$ALL_EPISODE_SCORES" | while read score; do
        if [ -n "$score" ]; then
            score_int=$(echo "$score" | cut -d. -f1)
            echo "  Episode $EPISODE_NUM: $score_int out of 25"
            EPISODE_NUM=$((EPISODE_NUM + 1))
        fi
    done
    echo ""
    echo "Average Score: $AVERAGE_INT out of 25"
    echo "Final Score: $FINAL_SCORE_INT out of 25"
    echo "Episodes Played: 3"
    echo "Players: $PLAYERS"
    echo ""
    echo "Logs saved to:"
    echo "  Setup: $SETUP_LOG"
    echo "  Game:  $GAME_LOG"
    echo "  Errors: $ERROR_LOG"
    if [ -s "$ERROR_LOG" ]; then
        echo ""
        echo "⚠️  Errors were logged. Check $ERROR_LOG for details."
    fi
} >> "$GAME_LOG"

# Display episode scores and average to terminal
echo "Episodes: $(echo "$ALL_EPISODE_SCORES" | tr '\n' ' ' | sed 's/ $//' | sed 's/\.[0-9]*//g') | Avg: $AVERAGE_INT out of 25" | tee -a "$GAME_LOG"