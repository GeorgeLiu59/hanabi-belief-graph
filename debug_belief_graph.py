#!/usr/bin/env python3
"""
Belief Graph Debug Helper - Interactive Log Analyzer

Helps quickly analyze belief graph logs for:
- Player identity and card information
- Agent actions and reasoning
- Belief updates and LLM prompts
"""

import os
import re
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class BeliefGraphDebugger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.current_log = None
        self.log_content = ""
        self.lines = []

    def list_logs(self) -> List[str]:
        """List all available log files."""
        if not self.log_dir.exists():
            print(f"❌ Log directory '{self.log_dir}' not found!")
            return []

        # Prioritize agent logs over game logs
        agent_logs = sorted(list(self.log_dir.glob("agent_*.log")), key=lambda x: x.stat().st_mtime, reverse=True)
        game_logs = sorted(list(self.log_dir.glob("game_*.log")), key=lambda x: x.stat().st_mtime, reverse=True)
        other_logs = sorted(list(self.log_dir.glob("*.log")), key=lambda x: x.stat().st_mtime, reverse=True)

        # Combine and deduplicate
        all_logs = agent_logs + game_logs + other_logs
        seen = set()
        unique_logs = []
        for log in all_logs:
            if log.name not in seen:
                seen.add(log.name)
                unique_logs.append(log.name)

        return unique_logs

    def load_log(self, log_name: str) -> bool:
        """Load a specific log file."""
        log_path = self.log_dir / log_name
        if not log_path.exists():
            print(f"❌ Log file '{log_name}' not found!")
            return False

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                self.log_content = f.read()
                self.lines = self.log_content.split('\n')
                self.current_log = log_name
                print(f"✅ Loaded log: {log_name} ({len(self.lines):,} lines)")
                return True
        except Exception as e:
            print(f"❌ Error loading log: {e}")
            return False

    def search_player_identity(self) -> List[Tuple[str, str]]:
        """Find player identity information."""
        results = []

        for i, line in enumerate(self.lines):
            if 'Info Type: PLAYER_IDENTITY' in line:
                # Get the next line which contains the actual player identity
                if i + 1 < len(self.lines):
                    next_line = self.lines[i + 1]
                    if 'Info Message:' in next_line:
                        # Extract player identity from the message
                        player_info = next_line.split('Info Message:')[1].strip()
                        results.append((f"Line {i+2}", f"👤 Player Identity: {player_info}"))

        return results

    def search_hand_updates(self) -> List[Tuple[str, str]]:
        """Find hand updates showing what cards players have."""
        results = []

        for i, line in enumerate(self.lines):
            if 'Updated P' in line and 'hand:' in line:
                # Get the complete context around hand updates
                context_start = max(0, i - 2)
                context_end = min(len(self.lines), i + 5)
                context_lines = self.lines[context_start:context_end]

                # Extract useful information
                hand_info = []
                for ctx_line in context_lines:
                    if 'Updated P' in ctx_line and 'hand:' in ctx_line:
                        hand_info.append(f"🃏 {ctx_line.strip()}")
                    elif 'Before:' in ctx_line:
                        hand_info.append(f"📋 Before: {ctx_line.strip()}")
                    elif 'After:' in ctx_line:
                        hand_info.append(f"📋 After: {ctx_line.strip()}")
                    elif 'Card' in ctx_line and '→' in ctx_line:
                        hand_info.append(f"🔄 {ctx_line.strip()}")

                if hand_info:
                    results.append((f"Lines {context_start+1}-{context_end}", '\n'.join(hand_info)))

        return results

    def search_actions(self) -> List[Tuple[str, str]]:
        """Find agent actions."""
        results = []

        for i, line in enumerate(self.lines):
            if ('Action:' in line or 'Parsed Action:' in line) and '{' in line:
                # Extract the action JSON (might span multiple lines)
                action_start = line.find('{')
                if action_start != -1:
                    try:
                        # Get the complete JSON (might span multiple lines)
                        json_text = line[action_start:]
                        j = i + 1

                        # Keep reading lines until we find the closing brace
                        while j < len(self.lines) and '}' not in json_text:
                            json_text += '\n' + self.lines[j]
                            j += 1

                        action_data = json.loads(json_text)

                        # Format the action nicely
                        action_type = action_data.get('action_type', 'UNKNOWN')
                        details = []

                        if action_type == 'REVEAL_COLOR':
                            details.append(f"Color: {action_data.get('color')}")
                            details.append(f"Target: Player {action_data.get('target_offset') + 1}")
                        elif action_type == 'REVEAL_RANK':
                            details.append(f"Rank: {action_data.get('rank')}")
                            details.append(f"Target: Player {action_data.get('target_offset') + 1}")
                        elif action_type in ['PLAY', 'DISCARD']:
                            details.append(f"Card Index: {action_data.get('card_index')}")

                        formatted_action = f"🎯 {action_type.upper()}: {', '.join(details)}"
                        results.append((f"Lines {i+1}-{j}", formatted_action))

                    except json.JSONDecodeError:
                        # Fallback to showing raw action
                        action_preview = line.strip()
                        if len(action_preview) > 100:
                            action_preview = action_preview[:100] + "..."
                        results.append((f"Line {i+1}", f"🎯 Action: {action_preview}"))

        return results

    def search_agent_actions(self) -> List[Tuple[str, str]]:
        """Find agent actions and reasoning."""
        results = []
        action_blocks = []

        # Find all reasoning process blocks
        for i, line in enumerate(self.lines):
            if 'Reasoning Process:' in line:
                action_blocks.append((i, line))
            elif 'Quality Score:' in line and action_blocks:
                # Found end of a reasoning block
                start_idx, start_line = action_blocks[-1]
                action_blocks[-1] = (start_idx, i, start_line, line)

        # Extract action information
        for block in action_blocks:
            if len(block) == 4:
                start_idx, end_idx, start_line, end_line = block
                action_info = start_line.strip()

                # Get some context around the action
                context_start = max(0, start_idx - 3)
                context_end = min(len(self.lines), end_idx + 3)
                context = '\n'.join(self.lines[context_start:context_end])

                results.append((f"Lines {start_idx+1}-{end_idx+1}", f"{action_info}\n\nContext:\n{context}"))

        return results

    def search_belief_updates(self) -> List[Tuple[str, str]]:
        """Find belief update events."""
        results = []

        for i, line in enumerate(self.lines):
            if 'Processing immediate belief update:' in line:
                try:
                    # Extract the JSON event
                    json_start = line.find('{')
                    if json_start != -1:
                        try:
                            # Try to find complete JSON (might span multiple lines)
                            json_text = line[json_start:]
                            event_data = json.loads(json_text)

                            formatted_event = json.dumps(event_data, indent=2)
                            results.append((f"Line {i+1}", f"Belief Update Event:\n{formatted_event}"))
                        except json.JSONDecodeError:
                            # If JSON is incomplete, try to get more lines
                            full_json = line[json_start:]
                            j = i + 1
                            while j < len(self.lines) and '}' not in full_json:
                                full_json += '\n' + self.lines[j]
                                j += 1

                            try:
                                event_data = json.loads(full_json)
                                formatted_event = json.dumps(event_data, indent=2)
                                results.append((f"Lines {i+1}-{j}", f"Belief Update Event:\n{formatted_event}"))
                            except:
                                results.append((f"Line {i+1}", f"Belief Update: {line.strip()}"))
                    else:
                        results.append((f"Line {i+1}", line.strip()))

                except Exception as e:
                    results.append((f"Line {i+1}", f"Error parsing belief update: {line.strip()}"))

        return results

    def search_llm_prompts(self) -> List[Tuple[str, str]]:
        """Find LLM prompts sent for belief updates."""
        results = []
        prompt_blocks = []

        # Find prompt blocks
        for i, line in enumerate(self.lines):
            if 'PROMPT_TO_LLM' in line or 'BELIEF_UPDATE_PROMPT' in line:
                prompt_blocks.append(i)

        for prompt_start in prompt_blocks:
            prompt_lines = []
            i = prompt_start + 1

            # Collect prompt content until we hit a log line from a different category
            while i < len(self.lines):
                line = self.lines[i].strip()

                # Stop if we hit a new log category
                if line and not line.startswith(' ') and '[' in line and ']' in line:
                    break

                if line:
                    prompt_lines.append(line)
                i += 1

            if prompt_lines:
                prompt_text = '\n'.join(prompt_lines)
                results.append((f"Lines {prompt_start+1}-{i}", f"LLM Prompt:\n{prompt_text}"))

        return results

    def search_game_state_updates(self) -> List[Tuple[str, str]]:
        """Find game state tracker updates."""
        results = []

        for i, line in enumerate(self.lines):
            if 'EARLY_UPDATE' in line or 'EARLY_UPDATE_COMPLETE' in line:
                results.append((f"Line {i+1}", line.strip()))
            elif 'Game state tracker updated:' in line:
                results.append((f"Line {i+1}", line.strip()))
            elif 'Belief graph synced:' in line:
                results.append((f"Line {i+1}", line.strip()))

        return results

    def interactive_search(self):
        """Interactive search interface."""
        if not self.current_log:
            print("❌ No log file loaded!")
            return

        while True:
            print("\n" + "="*60)
            print(f"🔍 Current Log: {self.current_log}")
            print("="*60)
            print("Search Options:")
            print("1. Player Identity (PLAYER_IDENTITY)")
            print("2. Hand Updates (cards players have)")
            print("3. Agent Actions (Action: {...})")
            print("4. Agent Actions & Reasoning (if available)")
            print("5. Belief Updates (Processing immediate belief update)")
            print("6. LLM Prompts (PROMPT_TO_LLM)")
            print("7. Game State Updates (EARLY_UPDATE)")
            print("8. Custom Search (regex)")
            print("9. Switch Log File")
            print("0. Exit")
            print("-"*60)

            choice = input("Enter choice (0-9): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                self.display_results(self.search_player_identity())
            elif choice == '2':
                self.display_results(self.search_hand_updates())
            elif choice == '3':
                self.display_results(self.search_actions())
            elif choice == '4':
                self.display_results(self.search_agent_actions())
            elif choice == '5':
                self.display_results(self.search_belief_updates())
            elif choice == '6':
                self.display_results(self.search_llm_prompts())
            elif choice == '7':
                self.display_results(self.search_game_state_updates())
            elif choice == '8':
                pattern = input("Enter search pattern (regex): ").strip()
                self.custom_search(pattern)
            elif choice == '9':
                self.switch_log()
            else:
                print("❌ Invalid choice!")

    def display_results(self, results: List[Tuple[str, str]], max_results: int = 10):
        """Display search results with pagination."""
        if not results:
            print("🔍 No results found!")
            return

        total = len(results)
        print(f"\n🔍 Found {total} results (showing max {max_results} at a time):")
        print("="*80)

        page = 0
        page_size = max_results

        while page * page_size < total:
            start_idx = page * page_size
            end_idx = min(start_idx + page_size, total)

            for i in range(start_idx, end_idx):
                location, content = results[i]
                print(f"\n📍 {location}")
                print("-" * 40)
                print(content)
                print("-" * 40)

            if end_idx < total:
                action = input(f"\nShowing {start_idx+1}-{end_idx} of {total}. Press Enter for more, 'q' to quit: ").strip().lower()
                if action == 'q':
                    break
                page += 1
            else:
                input(f"\nShowing all {total} results. Press Enter to continue: ")
                break

    def custom_search(self, pattern: str):
        """Custom regex search."""
        try:
            regex = re.compile(pattern, re.IGNORECASE)
            results = []

            for i, line in enumerate(self.lines):
                if regex.search(line):
                    results.append((f"Line {i+1}", line.strip()))

            self.display_results(results, max_results=20)

        except re.error as e:
            print(f"❌ Invalid regex pattern: {e}")

    def switch_log(self):
        """Switch to a different log file."""
        logs = self.list_logs()
        if not logs:
            return

        print("\nAvailable log files:")
        for i, log in enumerate(logs[:10], 1):  # Show latest 10
            print(f"{i}. {log}")

        choice = input(f"\nEnter log number (1-{min(len(logs), 10)}): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(logs):
                self.load_log(logs[idx])
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Invalid input!")

def main():
    """Main entry point."""
    print("🎮 Belief Graph Debug Helper")
    print("="*50)

    # Check if log directory exists
    log_dirs = ["logs", ".", ".."]
    log_dir = None

    for directory in log_dirs:
        if Path(directory).exists() and any(Path(directory).glob("*.log")):
            log_dir = directory
            break

    if not log_dir:
        print("❌ No log files found! Please run from a directory with log files.")
        return

    debugger = BeliefGraphDebugger(log_dir)

    # Auto-load latest log
    logs = debugger.list_logs()
    if logs:
        print(f"📁 Using log directory: {log_dir}")
        debugger.load_log(logs[0])
        debugger.interactive_search()
    else:
        print("❌ No log files found!")

if __name__ == "__main__":
    main()