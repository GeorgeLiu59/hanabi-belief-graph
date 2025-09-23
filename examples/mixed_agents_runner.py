#!/usr/bin/env python3
"""Runner for mixed agent types in Hanabi."""

import sys
import os
import getopt
import atexit
import signal
from typing import Dict, List, Any, Optional

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hanabi_learning_environment import rl_env
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent
from hanabi_learning_environment.agents.gemini_agent import GeminiAgent
from hanabi_learning_environment.agents.belief_graph_agents import (
    BeliefGraphCertaintyAgent,
    BeliefGraphProbabilisticAgent,
    BeliefGraphToMAgent
)

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

class MixedAgentsRunner:
    """Runner for games with different agent types."""
    
    def __init__(self, players: int, episodes: int, agent_configs: List[Dict[str, str]]):
        """Initialize the runner.
        
        Args:
            players: Number of players
            episodes: Number of episodes to run
            agent_configs: List of agent configurations, one per player
        """
        self.players = players
        self.episodes = episodes
        self.agent_configs = agent_configs
        self.environment = rl_env.make('Hanabi-Full-CardKnowledge', num_players=players)
        
        # Validate we have the right number of agent configs
        if len(agent_configs) != players:
            raise ValueError(f"Need {players} agent configs, got {len(agent_configs)}")
    
    def create_agents(self, episode_num: int = 0) -> List:
        """Create agents based on configuration."""
        agents = []
        base_config = {'players': self.players}
        
        for i, agent_config in enumerate(self.agent_configs):
            agent_class_name = agent_config['type']
            
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
            print(f"Created Player {i+1}: {agent_class_name}" + 
                  (f" ({agent_config.get('strategy', '')})" if agent_class_name == 'BeliefGraphAgent' else ""))
        
        return agents
    
    def run_episode(self, episode_num: int = 0) -> float:
        """Run a single episode."""
        observations = self.environment.reset()
        agents = self.create_agents(episode_num)
        
        done = False
        episode_reward = 0
        turn_count = 0
        
        print(f"\n{'='*60}")
        print(f"EPISODE {episode_num + 1} START")
        print(f"{'='*60}")
        
        while not done:
            turn_count += 1
            current_player_action = None
            
            # Get actions from all agents
            for agent_id, agent in enumerate(agents):
                observation = observations['player_observations'][agent_id]
                
                if observation['current_player'] == agent_id:
                    print(f"\n--- Turn {turn_count}: Player {agent_id + 1} ({type(agent).__name__}) ---")
                    print(f"Information tokens: {observation['information_tokens']}/8")
                    print(f"Life tokens: {observation['life_tokens']}/3")
                    print(f"Fireworks: {observation['fireworks']}")
                    
                    action = agent.act(observation)
                    print(f"Action: {action}")
                    current_player_action = action
                else:
                    action = agent.act(observation)
                    assert action is None, f"Agent {agent_id} should not act on other's turn"
            
            # Execute the current player's action
            if current_player_action is not None:
                observations, reward, done, _ = self.environment.step(current_player_action)
                episode_reward += reward
            else:
                raise ValueError("No valid action found for current player")
        
        print(f"\nEpisode {episode_num + 1} completed with reward: {episode_reward}")
        return episode_reward
    
    def run(self) -> List[float]:
        """Run multiple episodes."""
        rewards = []
        
        print(f"Starting {self.episodes} episodes with {self.players} players")
        print("Agent configuration:")
        for i, config in enumerate(self.agent_configs):
            agent_desc = config['type']
            if config['type'] == 'BeliefGraphAgent':
                agent_desc += f" ({config.get('strategy', 'probabilistic')})"
            print(f"  Player {i+1}: {agent_desc}")
        
        try:
            for episode in range(self.episodes):
                episode_reward = self.run_episode(episode)
                rewards.append(episode_reward)
                
                print(f'Episode {episode + 1}/{self.episodes} completed - Reward: {episode_reward}')
                if rewards:
                    print(f'Best so far: {max(rewards)} | Average: {sum(rewards)/len(rewards):.1f}')
                
        finally:
            self.cleanup()
        
        return rewards
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'environment'):
                del self.environment
            import gc
            gc.collect()
        except Exception:
            pass

def parse_agent_config(config_string: str) -> List[Dict[str, str]]:
    """Parse agent configuration string.
    
    Format: "GeminiAgent,BeliefGraphAgent:certainty,SimpleAgent"
    """
    configs = []
    for agent_spec in config_string.split(','):
        if ':' in agent_spec:
            agent_type, strategy = agent_spec.split(':', 1)
            configs.append({'type': agent_type, 'strategy': strategy})
        else:
            configs.append({'type': agent_spec})
    return configs

def main():
    """Main function."""
    if len(sys.argv) != 4:
        print("Usage: mixed_agents_runner.py <players> <episodes> <agent_config>")
        print("Example: mixed_agents_runner.py 2 3 'GeminiAgent,BeliefGraphAgent:certainty'")
        sys.exit(1)
    
    try:
        players = int(sys.argv[1])
        episodes = int(sys.argv[2])
        agent_config_string = sys.argv[3]
        
        agent_configs = parse_agent_config(agent_config_string)
        
        if len(agent_configs) != players:
            print(f"Error: Need {players} agent configurations, got {len(agent_configs)}")
            sys.exit(1)
        
        runner = MixedAgentsRunner(players, episodes, agent_configs)
        rewards = runner.run()
        
        # Final summary
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Episodes played: {len(rewards)}")
        print(f"Total reward: {sum(rewards)}")
        print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
        print(f"Best reward: {max(rewards)}")
        print(f"Episode rewards: {rewards}")
        
    except KeyboardInterrupt:
        print("\nGame interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        cleanup_resources()

if __name__ == "__main__":
    main()