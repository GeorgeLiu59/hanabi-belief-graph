This is not an officially supported Google product.

hanabi\_learning\_environment is a research platform for Hanabi experiments. The file rl\_env.py provides an RL environment using an API similar to OpenAI Gym. A lower level game interface is provided in pyhanabi.py for non-RL methods like Monte Carlo tree search.

### Getting started
Install the learning environment:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
pip install .                       # or pip install git+repo_url to install directly from github
```
Run the examples:
```
pip install numpy                   # game_example.py uses numpy
python examples/rl_env_example.py   # Runs RL episodes
python examples/game_example.py     # Plays a game using the lower level interface
```
Run the trained rainbow agent weights:
```
cd hanabi_learning_environment/agents/rainbow

python -um evaluate \
  --checkpoint_dir=checkpoints/checkpoints \
  --checkpoint_file_prefix=ckpt \
  --checkpoint_iteration=10000 \
  --gin_files='configs/hanabi_rainbow.gin' \
  --num_evaluation_games=100
```
(You can chance which iteration you are running 1000,2000,...,10000)


### New Contributions

**GeminiAgent**: A new agent implementation that uses Google's Gemini 2.0 Flash Lite language model to play Hanabi without any help or hints. The agent converts game state observations into natural language prompts, sends them to the LLM for reasoning, and parses JSON responses to extract actions. This approach tests how well language models can perform strategic reasoning in cooperative games with imperfect information, providing insights into LLM capabilities for multi-agent coordination without explicit training.
