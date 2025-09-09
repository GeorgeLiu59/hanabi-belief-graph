# coding=utf-8
"""Evaluation script for running a trained Rainbow agent on Hanabi."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from third_party.dopamine import checkpointer
import run_experiment
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_multi_string(
    'gin_files', [],
    'List of paths to gin configuration files (e.g.'
    '"configs/hanabi_rainbow.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1").')

flags.DEFINE_string('checkpoint_dir', None,
                    'Directory containing the checkpoint files.')
flags.DEFINE_string('checkpoint_file_prefix', 'ckpt',
                    'Prefix to use for the checkpoint files.')
flags.DEFINE_integer('checkpoint_iteration', None,
                     'Specific checkpoint iteration to load. If None, loads latest.')
flags.DEFINE_integer('num_evaluation_games', 100,
                     'Number of games to play for evaluation.')
flags.DEFINE_string('output_file', '',
                    'File to save evaluation results. If empty, prints to stdout.')


def load_checkpoint(agent, checkpoint_dir, checkpoint_file_prefix, iteration=None):
  """Load a checkpoint for the agent.
  
  Args:
    agent: The agent to load the checkpoint into.
    checkpoint_dir: Directory containing checkpoint files.
    checkpoint_file_prefix: Prefix for checkpoint files.
    iteration: Specific iteration to load. If None, loads latest.
    
  Returns:
    The iteration number that was loaded.
  """
  experiment_checkpointer = checkpointer.Checkpointer(
      checkpoint_dir, checkpoint_file_prefix)
  
  if iteration is None:
    # Load the latest checkpoint
    latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(
        checkpoint_dir)
    if latest_checkpoint_version < 0:
      raise ValueError('No checkpoints found in {}'.format(checkpoint_dir))
    iteration = latest_checkpoint_version
  
  # Load the specific checkpoint
  dqn_dictionary = experiment_checkpointer.load_checkpoint(iteration)
  if not agent.unbundle(checkpoint_dir, iteration, dqn_dictionary):
    raise ValueError('Failed to load checkpoint at iteration {}'.format(iteration))
  
  tf.logging.info('Loaded checkpoint from iteration %d', iteration)
  return iteration


def evaluate_agent(agent, environment, obs_stacker, num_games):
  """Evaluate the agent by playing multiple games.
  
  Args:
    agent: The trained agent.
    environment: The Hanabi environment.
    obs_stacker: Observation stacker object.
    num_games: Number of games to play.
    
  Returns:
    List of (episode_length, episode_return) tuples.
  """
  # Set agent to evaluation mode
  agent.eval_mode = True
  
  episode_data = []
  for game_idx in range(num_games):
    episode_length, episode_return = run_experiment.run_one_episode(
        agent, environment, obs_stacker)
    episode_data.append((episode_length, episode_return))
    
    if (game_idx + 1) % 10 == 0:
      tf.logging.info('Completed %d/%d games', game_idx + 1, num_games)
  
  return episode_data


def main(unused_argv):
  """Main evaluation function."""
  if FLAGS.checkpoint_dir is None:
    raise ValueError('--checkpoint_dir must be provided')
  
  # Load gin configuration
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  
  # Create environment and agent
  environment = run_experiment.create_environment()
  obs_stacker = run_experiment.create_obs_stacker(environment)
  agent = run_experiment.create_agent(environment, obs_stacker)
  
  # Load checkpoint
  loaded_iteration = load_checkpoint(
      agent, 
      FLAGS.checkpoint_dir, 
      FLAGS.checkpoint_file_prefix,
      FLAGS.checkpoint_iteration)
  
  # Run evaluation
  tf.logging.info('Starting evaluation with %d games...', FLAGS.num_evaluation_games)
  episode_data = evaluate_agent(agent, environment, obs_stacker, FLAGS.num_evaluation_games)
  
  # Calculate statistics
  episode_lengths, episode_returns = zip(*episode_data)
  avg_length = np.mean(episode_lengths)
  avg_return = np.mean(episode_returns)
  std_return = np.std(episode_returns)
  min_return = np.min(episode_returns)
  max_return = np.max(episode_returns)
  
  # Prepare results
  results = {
      'checkpoint_iteration': loaded_iteration,
      'num_games': FLAGS.num_evaluation_games,
      'average_episode_length': avg_length,
      'average_return': avg_return,
      'std_return': std_return,
      'min_return': min_return,
      'max_return': max_return,
      'episode_returns': episode_returns,
      'episode_lengths': episode_lengths
  }
  
  # Output results
  output_text = """
Evaluation Results:
==================
Checkpoint iteration: {checkpoint_iteration}
Number of games: {num_games}
Average episode length: {average_episode_length:.2f}
Average return: {average_return:.2f}
Standard deviation: {std_return:.2f}
Min return: {min_return:.2f}
Max return: {max_return:.2f}

Individual game results:
""".format(**results)
  
  for i, (length, ret) in enumerate(episode_data):
    output_text += "Game {}: length={}, return={:.2f}\n".format(i+1, length, ret)
  
  if FLAGS.output_file:
    with open(FLAGS.output_file, 'w') as f:
      f.write(output_text)
    tf.logging.info('Results saved to %s', FLAGS.output_file)
  else:
    print(output_text)


if __name__ == '__main__':
  app.run(main)
