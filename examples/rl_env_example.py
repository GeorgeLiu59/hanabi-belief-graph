# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple episode runner using the RL environment for Hanabi."""

import sys
import getopt
import atexit
import signal
from typing import Dict, List, Any

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.gemini_agent import GeminiAgent

# Available agent classes
AGENT_CLASSES = {
    'SimpleAgent': SimpleAgent, 
    'RandomAgent': RandomAgent, 
    'GeminiAgent': GeminiAgent
}

# Default configuration
DEFAULT_CONFIG = {
    'players': 2,
    'num_episodes': 1,
    'agent_class': 'GeminiAgent'
}


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


class HanabiRunner:
    """Runner class for Hanabi episodes."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the runner.
        
        Args:
            config: Dictionary containing game configuration parameters.
        """
        self.config = config
        self.agent_config = {'players': config['players']}
        self.environment = rl_env.make('Hanabi-Full-CardKnowledge', num_players=config['players'])
        self.agent_class = AGENT_CLASSES[config['agent_class']]

    def run_episode(self) -> float:
        """Run a single episode.
        
        Returns:
            float: The total reward for this episode.
        """
        observations = self.environment.reset()
        agents = [
            self.agent_class(self.agent_config) 
            for _ in range(self.config['players'])
        ]
        
        done = False
        episode_reward = 0
        
        while not done:
            current_player_action = None
            
            # Get actions from all agents
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                action = agent.act(observation)
                
                if observation['current_player'] == agent_id:
                    assert action is not None, f"Agent {agent_id} must provide action on their turn"
                    current_player_action = action
                else:
                    assert action is None, f"Agent {agent_id} should not act on other's turn"
            
            # Execute the current player's action
            if current_player_action is not None:
                print(f'Agent: {observations["current_player"]} action: {current_player_action}')
                observations, reward, done, _ = self.environment.step(current_player_action)
                episode_reward += reward
            else:
                raise ValueError("No valid action found for current player")
        
        return episode_reward

    def run(self) -> List[float]:
        """Run multiple episodes.
        
        Returns:
            List[float]: List of rewards for each episode.
        """
        rewards = []
        
        try:
            for episode in range(self.config['num_episodes']):
                episode_reward = self.run_episode()
                rewards.append(episode_reward)
                
                print(f'Running episode: {episode}')
                print(f'Max Reward: {max(rewards):.3f}')
                
        finally:
            self.cleanup()
            
        return rewards

    def cleanup(self):
        """Clean up resources to prevent C++ cleanup errors."""
        try:
            # Clear references to allow garbage collection
            if hasattr(self, 'environment'):
                del self.environment
            if hasattr(self, 'agent_class'):
                del self.agent_class
            if hasattr(self, 'agent_config'):
                del self.agent_config
            # Force garbage collection
            import gc
            gc.collect()
        except Exception:
            pass


def parse_command_line_args() -> Dict[str, Any]:
    """Parse command line arguments.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()
    
    try:
        options, arguments = getopt.getopt(
            sys.argv[1:], 
            '', 
            ['players=', 'num_episodes=', 'agent_class=']
        )
        
        if arguments:
            available_agents = ' or '.join(AGENT_CLASSES.keys())
            sys.exit(
                f'Usage: rl_env_example.py [options]\n'
                f'--players       number of players in the game.\n'
                f'--num_episodes  number of game episodes to run.\n'
                f'--agent_class   {available_agents}'
            )
        
        for flag, value in options:
            flag = flag[2:]  # Strip leading --
            if flag in config:
                config[flag] = type(config[flag])(value)
            else:
                print(f"Warning: Unknown flag '{flag}' ignored")
                
    except getopt.GetoptError as e:
        print(f"Error parsing arguments: {e}")
        sys.exit(1)
    
    return config


def main():
    """Main function to run Hanabi episodes."""
    try:
        # Parse command line arguments
        config = parse_command_line_args()
        
        # Create and run the game
        runner = HanabiRunner(config)
        runner.run()
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error running game: {e}")
        sys.exit(1)
    finally:
        # Final cleanup
        cleanup_resources()


if __name__ == "__main__":
    main()
