#!/usr/bin/env python3
"""Async parallel game runner for Hanabi."""

import sys
import os
import asyncio
import atexit
import signal
from typing import Dict, List, Any, Optional
from datetime import datetime
import threading

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.gemini_agent import GeminiAgent
from hanabi_learning_environment.agents.certainty_agent import BeliefGraphCertaintyAgent
from hanabi_learning_environment.agents.probabilistic_agent import BeliefGraphProbabilisticAgent
from hanabi_learning_environment.agents.tom_agent import BeliefGraphToMAgent
from hanabi_learning_environment.game_logger import TerminalLogger, GameStateLogger, EventLogger

# Available agent classes
AGENT_CLASSES = {
    'SimpleAgent': SimpleAgent,
    'RandomAgent': RandomAgent,
    'GeminiAgent': GeminiAgent,
    'BeliefGraphAgent': {  # Map strategies to specific classes
        'certainty': BeliefGraphCertaintyAgent,
        'probabilistic': BeliefGraphProbabilisticAgent,
        'theory_of_mind': BeliefGraphToMAgent
    }
}

# Global lock for thread-safe operations
_global_lock = threading.Lock()


def cleanup_resources():
    """Clean up resources to prevent C++ cleanup errors."""
    try:
        import gc
        gc.collect()
    except Exception:
        pass


def signal_handler(signum, frame):
    """Handle signals for clean shutdown."""
    cleanup_resources()
    sys.exit(0)


# Register cleanup functions
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class AsyncGameRunner:
    """Async runner for a single Hanabi game.

    Logging Structure:
    - Agent logs: logs/MODEL_MODE/YYYYMMDD_HHMMSS_MODEL_MODE.log
      Examples:
        - logs/bg_probabilistic/20260102_230456_BG_probabilistic.log
        - logs/gemini_base/20260102_230500_Gemini_base.log
        - logs/bg_certainty/20260102_230600_BG_certainty.log

    - Game state logs: logs/game_state/YYYYMMDD_HHMMSS_AGENT1-AGENT2/
      Examples:
        - logs/game_state/20260102_230456_BG_probabilistic-Simple/

    - Event logs: logs/events/YYYYMMDD_HHMMSS_AGENT1-AGENT2/
      Examples:
        - logs/events/20260102_230456_BG_probabilistic-Random/
    """

    def __init__(
        self,
        game_id: int,
        players: int,
        agent_configs: List[Dict[str, str]],
        max_turns: Optional[int] = None,
        log_prefix: str = "",
        agent_log_dir: str = "logs"
    ):
        """Initialize the async game runner.

        Args:
            game_id: Unique identifier for this game
            players: Number of players
            agent_configs: List of agent configurations, one per player
            max_turns: Maximum number of turns (None for no limit)
            log_prefix: Prefix for log files
            agent_log_dir: Directory for agent logs (default: "logs")
        """
        self.game_id = game_id
        self.players = players
        self.agent_configs = agent_configs
        self.max_turns = max_turns
        self.log_prefix = log_prefix
        self.agent_log_dir = agent_log_dir

        # Create environment (each game needs its own)
        self.environment = rl_env.make('Hanabi-Full-CardKnowledge', num_players=players)

        # Build log directory name from agent configs
        # Format: DATE_TIME_MODEL_MODE (or DATE_TIME_MODEL1-MODEL2 for multiple agents)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Extract model types and modes from agent configs
        agent_names = []
        for config in agent_configs:
            agent_type = config['type']
            if agent_type == 'BeliefGraphAgent':
                mode = config.get('strategy', 'probabilistic')
                agent_names.append(f"BG_{mode}")
            elif agent_type == 'GeminiAgent':
                agent_names.append("Gemini_base")
            elif agent_type == 'SimpleAgent':
                agent_names.append("Simple")
            elif agent_type == 'RandomAgent':
                agent_names.append("Random")
            else:
                agent_names.append(agent_type)

        # Create directory name: DATE_TIME_AGENT1-AGENT2
        log_suffix = "-".join(agent_names)
        log_dir_name = f"{timestamp}_{log_suffix}"

        # Initialize logging with new naming format
        # Note: Agent logs (DATE_TIME_MODEL_MODE.log) are created by agents themselves
        self.game_state_logger = GameStateLogger(
            log_dir=f"logs/game_state/{log_dir_name}",
            game_id=game_id
        )
        self.event_logger = EventLogger(
            log_dir=f"logs/events/{log_dir_name}",
            game_id=game_id
        )

        # Validate we have the right number of agent configs
        if len(agent_configs) != players:
            raise ValueError(f"Need {players} agent configs, got {len(agent_configs)}")

    def create_agents(self, episode_num: int = 0) -> List:
        """Create agents based on configuration."""
        agents = []

        for i, agent_config in enumerate(self.agent_configs):
            agent_class_name = agent_config['type']

            # Create base config with tag and game_id
            base_config = {
                'players': self.players,
                'tag': agent_config.get('tag', ''),
                'game_id': self.game_id
            }

            # Handle BeliefGraphAgent which maps to different classes based on strategy
            if agent_class_name == 'BeliefGraphAgent':
                strategy = agent_config.get('strategy', 'probabilistic')
                agent_class = AGENT_CLASSES[agent_class_name][strategy]
                agent = agent_class(base_config)
            else:
                agent_class = AGENT_CLASSES[agent_class_name]
                agent = agent_class(base_config)

            # Set episode number and reset
            if hasattr(agent, 'set_episode'):
                agent.set_episode(episode_num)
            if hasattr(agent, 'reset'):
                agent.reset(base_config)

            agents.append(agent)

        return agents

    def _extract_fireworks_total(self, observation: Dict[str, Any]) -> int:
        """Compute total fireworks built from an observation."""
        player_observations = observation.get('player_observations', [])
        if not player_observations:
            return 0
        fireworks = player_observations[0].get('fireworks', {})
        return sum(fireworks.values())

    async def run_game_async(self, episode_num: int = 0) -> Dict[str, Any]:
        """Run a single game asynchronously.

        This wraps the synchronous game execution in an async context
        to allow concurrent execution of multiple games.
        """
        # Run in executor to not block the event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._run_game_sync, episode_num)

    def _run_game_sync(self, episode_num: int = 0) -> Dict[str, Any]:
        """Run a single game synchronously (called from async wrapper)."""
        observations = self.environment.reset()
        agents = self.create_agents(episode_num)

        # Log episode start
        self.game_state_logger.log_episode_start(episode_num)
        self.event_logger.log_episode_start(episode_num)

        done = False
        episode_reward = 0
        turn_count = 0
        final_fireworks = self._extract_fireworks_total(observations)
        max_fireworks = final_fireworks

        print(f"\n[Game {self.game_id}] Episode {episode_num + 1} START")

        while not done and (self.max_turns is None or turn_count < self.max_turns):
            turn_count += 1
            current_player_action = None
            current_player_id = None

            # Get actions from all agents
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]

                if observation['current_player'] == agent_id:
                    current_player_id = agent_id
                    print(f"[Game {self.game_id}] Turn {turn_count}: Player {agent_id + 1} ({type(agent).__name__})")
                    print(f"  Info: {observation['information_tokens']}/8, Lives: {observation['life_tokens']}/3, Fireworks: {observation['fireworks']}")

                    # Log state BEFORE action
                    self.game_state_logger.log_turn_state(observation, turn_count, agent_id, action=None)

                    # Try to get action from agent with error handling
                    try:
                        action = agent.act(observation)
                    except Exception as e:
                        print(f"[Game {self.game_id}] ERROR: Agent {agent_id} failed to produce action: {e}")
                        print(f"[Game {self.game_id}] Using random legal move as emergency fallback")
                        # Emergency fallback: pick random legal move
                        import random
                        legal_moves = observation.get('legal_moves', [])
                        if legal_moves:
                            action = random.choice(legal_moves)
                        else:
                            # No legal moves available, game should end
                            print(f"[Game {self.game_id}] CRITICAL: No legal moves available, ending game")
                            done = True
                            break

                    print(f"  Action: {action}")
                    current_player_action = action
                else:
                    action = agent.act(observation)
                    assert action is None, f"Agent {agent_id} should not act on other's turn"

            # Execute the current player's action
            if current_player_action is not None:
                observations, reward, done, _ = self.environment.step(current_player_action)
                episode_reward += reward
                final_fireworks = self._extract_fireworks_total(observations)
                if final_fireworks > max_fireworks:
                    max_fireworks = final_fireworks

                # Log state AFTER action and check for events
                new_observation = observations['player_observations'][current_player_id]
                self.event_logger.check_and_log_changes(new_observation, turn_count, current_player_id, current_player_action)
            else:
                raise ValueError("No valid action found for current player")

        # Determine end reason
        final_obs = observations['player_observations'][0]
        if self.max_turns is not None and turn_count >= self.max_turns:
            end_reason = "max_turns"
        elif final_obs.get('life_tokens', 0) == 0:
            end_reason = "lives_depleted"
        elif final_fireworks == 25:
            end_reason = "perfect_score"
        elif final_obs.get('deck_size', 0) == 0:
            end_reason = "deck_exhausted"
        else:
            end_reason = "max_turns"

        # Log episode end
        self.game_state_logger.log_episode_end(episode_num, final_fireworks, max_fireworks, turn_count)
        self.event_logger.log_episode_end(episode_num, end_reason)

        print(f"[Game {self.game_id}] Episode {episode_num + 1} COMPLETE - Reward: {episode_reward}, Fireworks: {final_fireworks}/{max_fireworks}, Turns: {turn_count}, Reason: {end_reason}")

        return {
            'game_id': self.game_id,
            'episode': episode_num,
            'reward': episode_reward,
            'final_fireworks': final_fireworks,
            'max_fireworks': max_fireworks,
            'end_reason': end_reason,
            'turns': turn_count
        }

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'environment'):
                del self.environment
            import gc
            gc.collect()
        except Exception:
            pass


class ParallelGameCoordinator:
    """Coordinates multiple games running in parallel."""

    def __init__(self):
        """Initialize the coordinator."""
        self.results: List[Dict[str, Any]] = []

    async def run_games_parallel(
        self,
        num_games: int,
        players: int,
        agent_configs: List[Dict[str, str]],
        max_turns: Optional[int] = None,
        episodes_per_game: int = 1,
        max_concurrent: int = 10
    ) -> List[Dict[str, Any]]:
        """Run multiple games in parallel with concurrency control.

        Args:
            num_games: Number of games to run total
            players: Number of players per game
            agent_configs: Agent configurations
            max_turns: Maximum turns per game
            episodes_per_game: Episodes to run per game
            max_concurrent: Maximum concurrent games (default: 10, respects Gemini API limit)

        Returns:
            List of result dictionaries from all games
        """
        print(f"\n{'='*80}")
        print(f"Starting {num_games} games with {players} players each")
        print(f"Max concurrent games: {max_concurrent}")
        print(f"Max turns per game: {max_turns if max_turns else 'unlimited'}")
        print(f"Episodes per game: {episodes_per_game}")
        print(f"{'='*80}\n")

        # Create game runners
        runners = []
        for game_id in range(num_games):
            runner = AsyncGameRunner(
                game_id=game_id,
                players=players,
                agent_configs=agent_configs,
                max_turns=max_turns
            )
            runners.append(runner)

        # Run games in batches to respect concurrency limit
        all_results = []

        for batch_start in range(0, num_games, max_concurrent):
            batch_end = min(batch_start + max_concurrent, num_games)
            batch_runners = runners[batch_start:batch_end]

            print(f"\n{'='*80}")
            print(f"Launching batch: Games {batch_start} to {batch_end-1} ({len(batch_runners)} games)")
            print(f"{'='*80}\n")

            # Create tasks for this batch
            tasks = []
            for runner in batch_runners:
                for episode in range(episodes_per_game):
                    task = runner.run_game_async(episode)
                    tasks.append(task)

            # Wait for batch to complete
            print(f"Running {len(tasks)} game tasks in this batch...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)

            print(f"\nBatch {batch_start//max_concurrent + 1} complete!")

        results = all_results

        # Filter out exceptions and collect results
        self.results = []
        failed_games = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_games += 1
                print(f"\n[ERROR] Game task {i} failed with exception: {result}")
                import traceback
                traceback.print_exception(type(result), result, result.__traceback__)
            else:
                self.results.append(result)

        if failed_games > 0:
            print(f"\n[WARNING] {failed_games} out of {len(results)} game tasks failed")

        # Cleanup
        for runner in runners:
            runner.cleanup()

        print(f"\n{'='*80}")
        print(f"All {num_games} games completed!")
        print(f"{'='*80}\n")

        self._print_summary()

        return self.results

    def _print_summary(self):
        """Print summary of all game results."""
        if not self.results:
            print("No results to summarize.")
            return

        print("\n" + "="*80)
        print("SUMMARY OF ALL GAMES")
        print("="*80)

        # Group by game_id
        games = {}
        for result in self.results:
            game_id = result['game_id']
            if game_id not in games:
                games[game_id] = []
            games[game_id].append(result)

        # Print per-game summary
        for game_id in sorted(games.keys()):
            episodes = games[game_id]
            print(f"\nGame {game_id}:")
            for ep in episodes:
                print(f"  Episode {ep['episode']}: Score={ep['final_fireworks']}, Turns={ep['turns']}, Reason={ep['end_reason']}")

        # Overall statistics
        total_fireworks = sum(r['final_fireworks'] for r in self.results)
        avg_fireworks = total_fireworks / len(self.results) if self.results else 0
        total_turns = sum(r['turns'] for r in self.results)
        avg_turns = total_turns / len(self.results) if self.results else 0

        print(f"\nOverall Statistics:")
        print(f"  Total games: {len(games)}")
        print(f"  Total episodes: {len(self.results)}")
        print(f"  Average fireworks: {avg_fireworks:.2f}")
        print(f"  Average turns: {avg_turns:.2f}")
        print("="*80 + "\n")


def parse_agent_config(config_string: str) -> List[Dict[str, str]]:
    """Parse agent configuration string.

    Format: "GeminiAgent,BeliefGraphAgent:certainty,SimpleAgent"
    """
    configs = []
    agent_strings = config_string.split(',')

    for agent_string in agent_strings:
        agent_string = agent_string.strip()
        if ':' in agent_string:
            agent_type, strategy = agent_string.split(':', 1)
            configs.append({'type': agent_type.strip(), 'strategy': strategy.strip()})
        else:
            configs.append({'type': agent_string})

    return configs


async def main_async():
    """Main async entry point."""
    # Example usage
    if len(sys.argv) < 5:
        print("Usage: python async_game_runner.py <num_games> <players> <episodes_per_game> <agent_config> [max_turns] [--tag TAG]")
        print("Example: python async_game_runner.py 3 2 1 'SimpleAgent,RandomAgent' 5")
        print("Example with tag: python async_game_runner.py 3 2 1 'BeliefGraphAgent:probabilistic,SimpleAgent' 5 --tag \"test run\"")
        sys.exit(1)

    num_games = int(sys.argv[1])
    players = int(sys.argv[2])
    episodes_per_game = int(sys.argv[3])
    agent_config_str = sys.argv[4]

    # Parse tag argument first
    tag = ""
    tag_index = -1
    for i, arg in enumerate(sys.argv):
        if arg == "--tag" and i + 1 < len(sys.argv):
            tag = sys.argv[i + 1]
            tag_index = i
            break

    # Parse max_turns (position 5, but only if it's not --tag)
    max_turns = None
    if len(sys.argv) > 5 and sys.argv[5] != "--tag":
        try:
            max_turns = int(sys.argv[5])
        except ValueError:
            pass  # Not a number, leave as None

    agent_configs = parse_agent_config(agent_config_str)

    # Add tag to all agent configs
    for config in agent_configs:
        config['tag'] = tag

    if len(agent_configs) != players:
        print(f"Error: Need {players} agent configs, got {len(agent_configs)}")
        sys.exit(1)

    # Create coordinator and run games
    # Limit to 10 concurrent games to respect Gemini API concurrent request limits
    coordinator = ParallelGameCoordinator()
    await coordinator.run_games_parallel(
        num_games=num_games,
        players=players,
        agent_configs=agent_configs,
        max_turns=max_turns,
        episodes_per_game=episodes_per_game,
        max_concurrent=10  # Gemini API concurrent request limit
    )


if __name__ == '__main__':
    asyncio.run(main_async())
