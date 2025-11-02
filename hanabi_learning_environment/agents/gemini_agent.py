"""Gemini LLM Agent for Hanabi with comprehensive logging and modular design."""

import os
import time
import random
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from hanabi_learning_environment.rl_env import Agent
from .agent_logger import AgentLogger
from .prompt_manager import PromptManager
from .action_validator import ActionValidator
from .cot_reasoning import ChainOfThoughtReasoner

# Load environment variables
load_dotenv()


class GeminiAgent(Agent):
    """Modular Gemini agent with comprehensive logging and Chain of Thought reasoning."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the Gemini agent with modular components."""
        self.config = config
        self.agent_id = str(random.randint(1000, 9999))
        
        # Initialize modular components
        self.logger = AgentLogger(self.agent_id)
        self.prompt_manager = PromptManager()
        self.action_validator = ActionValidator()
        self.cot_reasoner = ChainOfThoughtReasoner()
    
    def set_episode(self, episode_num: int):
        """Set current episode number for logging."""
        self.logger.set_episode(episode_num)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize API key
        self.api_key = self._load_api_key()
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Game state tracking
        self.current_game_id = None
        self.observation_history = []
        self.action_history = []
        self.history_size = 4

        # Rate limiting
        self.request_times = []
        self.max_requests_per_minute = 30
        self.current_minute = None
        
        self.logger.log_error("AGENT_INIT", f"Agent {self.agent_id} initialized successfully")
    
    def _load_api_key(self) -> str:
        """Load API key from environment or .env file."""
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            try:
                env_path = os.path.join(os.getcwd(), '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if line.startswith('GEMINI_API_KEY='):
                                api_key = line.split('=', 1)[1].strip().strip('"\'')
                                break
            except Exception as e:
                self.logger.log_error("API_KEY_LOAD", f"Could not read .env file: {e}")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment variables or .env file. "
                "Please set GEMINI_API_KEY environment variable or create a .env file "
                "in the project root with: GEMINI_API_KEY=your_key_here"
            )
        
        return api_key
        
    def reset(self, config):
        """Reset the agent for a new game."""
        self.config = config
        self.observation_history = []
        self.action_history = []
        self.current_game_id = random.randint(1000, 9999)
        self.logger.log_error("GAME_RESET", f"Agent {self.agent_id} reset for new game {self.current_game_id}")
        
    def _add_observation_to_history(self, observation: Dict[str, Any]):
        """Add current observation to history."""
        obs_summary = {
            'turn': len(self.observation_history) + 1,
            'information_tokens': observation['information_tokens'],
            'life_tokens': observation['life_tokens'],
            'deck_size': observation['deck_size'],
            'fireworks': observation['fireworks'].copy(),
            'discard_pile_size': len(observation['discard_pile']),
            'discard_pile': observation['discard_pile'].copy(),
            'my_hand': observation['observed_hands'][0].copy(),
            'my_knowledge': observation['card_knowledge'][0].copy(),
            'other_hands': [hand.copy() for hand in observation['observed_hands'][1:]],
            'legal_moves_count': len(observation['legal_moves'])
        }
        
        self.observation_history.append(obs_summary)
        if len(self.observation_history) > self.history_size:
            self.observation_history.pop(0)
    
    def _add_action_to_history(self, action: Dict[str, Any]):
        """Add the action taken to history."""
        self.action_history.append(action.copy())
        if len(self.action_history) > self.history_size:
            self.action_history.pop(0)
    
    def _check_rate_limit(self):
        """Check if we're at the rate limit and wait if necessary."""
        current_time = time.time()
        current_minute = int(current_time // 60)
        
        if self.current_minute != current_minute:
            self.request_times = []
            self.current_minute = current_minute
        
        if len(self.request_times) >= self.max_requests_per_minute:
            seconds_until_next_minute = 60 - (current_time % 60) + 1
            self.logger.log_error("RATE_LIMIT", f"Rate limit reached. Waiting {seconds_until_next_minute:.1f} seconds")
            time.sleep(seconds_until_next_minute)
            self.request_times = []
            self.current_minute = int(time.time() // 60)
        
        self.request_times.append(current_time)
    
    def _get_legal_action_with_retry(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """Get a legal action with comprehensive retry logic and logging."""
        start_time = time.time()
        
        # Create history and base prompt
        history = self.prompt_manager.format_history_for_llm(self.observation_history, self.action_history)
        variant = observation.get('belief_variant', 'certainty')
        base_prompt = self.prompt_manager.create_main_prompt(observation, history, variant)
        
        # Create Chain of Thought prompt
        cot_prompt = self.cot_reasoner.create_cot_prompt(observation, base_prompt)
        
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                # Check rate limits
                self._check_rate_limit()
                
                # Log prompt being sent
                prompt_to_send = cot_prompt if attempt == 0 else cot_prompt
                self.logger.log_prompt_sent(prompt_to_send, attempt + 1)
                
                # Get LLM response
                response = self.model.generate_content(prompt_to_send)
                response_text = response.text
                
                # Log response received
                self.logger.log_llm_response(response_text, attempt + 1)
                
                # Extract reasoning and action
                reasoning, action_json = self.cot_reasoner.extract_reasoning_and_action(response_text)
                
                # Log Chain of Thought reasoning
                if reasoning:
                    reasoning_quality = self.cot_reasoner.score_reasoning_quality(reasoning)
                    self.logger.log_chain_of_thought(f"Quality Score: {reasoning_quality:.2f}\n\n{reasoning}", {})
                
                # Parse action
                action = self.action_validator.parse_llm_response(action_json)
                self.logger.log_parsing_attempt(action_json, action)
                
                if action is None:
                    if attempt < max_retries:
                        error_msg = "Invalid JSON format. Please provide a valid JSON response."
                        cot_prompt = self.prompt_manager.create_retry_prompt(observation, error_msg)
                        self.logger.log_retry_attempt(attempt + 1, error_msg, cot_prompt)
                        continue
                    else:
                        self.logger.log_error("PARSING_FAILED", f"Failed to parse after {max_retries} retries")
                        break
                
                # Validate action completeness
                is_complete, completeness_msg = self.action_validator.validate_action_completeness(action)
                if not is_complete:
                    if attempt < max_retries:
                        cot_prompt = self.prompt_manager.create_retry_prompt(observation, completeness_msg)
                        self.logger.log_retry_attempt(attempt + 1, completeness_msg, cot_prompt)
                        continue
                    else:
                        self.logger.log_error("ACTION_INCOMPLETE", completeness_msg)
                        break
                
                # Validate action legality
                if action.get('action_type') == 'REVEAL_RANK' and action.get('rank') is not None:
                    try:
                        rank_val = int(action['rank'])
                        action['rank'] = rank_val - 1
                    except (ValueError, TypeError):
                        pass
                is_legal = self.action_validator.is_action_legal(action, observation['legal_moves'])
                
                if is_legal:
                    # Legal action found!
                    self.logger.log_validation_result(action, observation['legal_moves'], True)
                    if reasoning:
                        self.logger.log_chain_of_thought(reasoning, action)
                    
                    execution_time = time.time() - start_time
                    self.logger.log_final_decision(action, execution_time)
                    return action
                else:
                    # Action is illegal
                    illegal_reason = self.action_validator.get_illegal_reason(action, observation['legal_moves'])
                    self.logger.log_validation_result(action, observation['legal_moves'], False, illegal_reason)
                    
                    if attempt < max_retries:
                        cot_prompt = self.prompt_manager.create_retry_prompt(observation, f"ILLEGAL MOVE: {illegal_reason}")
                        self.logger.log_retry_attempt(attempt + 1, illegal_reason, cot_prompt)
                        continue
                    else:
                        self.logger.log_error("ILLEGAL_ACTION_FINAL", f"Action remained illegal after {max_retries} retries")
                        break
                        
            except Exception as e:
                error_msg = f"API error on attempt {attempt + 1}: {e}"
                self.logger.log_error("API_ERROR", error_msg)
                
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    break
        
        # All retries failed - system must work without fallback
        execution_time = time.time() - start_time
        self.logger.log_error("ALL_RETRIES_FAILED", f"Failed to get valid action after {max_retries} retries")
        raise ValueError(f"Gemini agent failed to produce valid action after {max_retries} retries")
    
    def act(self, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Act based on an observation using modular components."""
        # Always observe and learn, regardless of whose turn it is
        # Add observation to history to learn from other players' actions
        self._add_observation_to_history(observation)
        
        # If not our turn, just observe - don't take action but still learn
        if observation['current_player_offset'] != 0:
            return None
        
        # Log decision start
        self.logger.log_decision_start(observation)
        
        try:
            # Get legal action with retry logic
            action = self._get_legal_action_with_retry(observation)
            
            # Add action to history
            self._add_action_to_history(action)
            
            return action
            
        except Exception as e:
            self.logger.log_error("ACT_ERROR", f"Critical error in act method: {e}")
            
            # No emergency fallback - let it fail
            raise
