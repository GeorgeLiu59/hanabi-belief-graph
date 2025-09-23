"""Comprehensive logging module for Hanabi agents."""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional


class AgentLogger:
    """Handles all logging for agent operations with complete visibility."""
    
    def __init__(self, agent_id: str, log_dir: str = "logs"):
        self.agent_id = agent_id
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_episode = 0
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Single log file per agent
        self.log_file = f"{log_dir}/agent_{agent_id}_{self.session_id}.log"
        
    def _timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def _write_to_log(self, log_type: str, content: str):
        """Write content to unified log file with timestamp and type."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{self._timestamp()}] [EPISODE_{self.current_episode}] [{log_type}] {content}\n")
    
    def set_episode(self, episode_num: int):
        """Set current episode number for logging."""
        self.current_episode = episode_num
    
    def log_decision_start(self, observation: Dict[str, Any]):
        """Log the start of a decision-making process."""
        content = f"DECISION_START - Agent {self.agent_id}\n"
        content += f"Current Player: {observation.get('current_player')}\n"
        content += f"Player Offset: {observation.get('current_player_offset')}\n"
        content += f"Information Tokens: {observation.get('information_tokens')}\n"
        content += f"Life Tokens: {observation.get('life_tokens')}\n"
        content += f"Legal Moves Count: {len(observation.get('legal_moves', []))}\n"
        content += f"Legal Moves: {json.dumps(observation.get('legal_moves', []), indent=2)}"
        self._write_to_log("DECISION", content)
    
    def log_prompt_sent(self, prompt: str, attempt: int = 1):
        """Log prompt sent to LLM."""
        content = f"PROMPT_SENT - Agent {self.agent_id} - Attempt {attempt}\n"
        content += f"Prompt Length: {len(prompt)} characters\n"
        content += "=" * 50 + "\n"
        content += prompt
        content += "\n" + "=" * 50
        self._write_to_log("LLM", content)
    
    def log_llm_response(self, response: str, attempt: int = 1):
        """Log response received from LLM."""
        content = f"LLM_RESPONSE - Agent {self.agent_id} - Attempt {attempt}\n"
        content += f"Response Length: {len(response)} characters\n"
        content += "=" * 50 + "\n"
        content += response
        content += "\n" + "=" * 50
        self._write_to_log("LLM", content)
    
    def log_parsing_attempt(self, response: str, parsed_action: Optional[Dict[str, Any]]):
        """Log JSON parsing attempt."""
        content = f"PARSING_ATTEMPT - Agent {self.agent_id}\n"
        content += f"Raw Response: {response}\n"
        if parsed_action:
            content += f"Parsed Action: {json.dumps(parsed_action, indent=2)}"
        else:
            content += "Parsing FAILED - Invalid JSON"
        self._write_to_log("VALIDATION", content)
    
    def log_validation_result(self, action: Dict[str, Any], legal_moves: list, is_legal: bool, reason: str = ""):
        """Log action validation result."""
        content = f"VALIDATION_RESULT - Agent {self.agent_id}\n"
        content += f"Action: {json.dumps(action, indent=2)}\n"
        content += f"Is Legal: {is_legal}\n"
        if not is_legal:
            content += f"Reason: {reason}\n"
        content += f"Available Legal Moves: {json.dumps(legal_moves, indent=2)}"
        self._write_to_log("VALIDATION", content)
    
    def log_retry_attempt(self, attempt: int, error_message: str, retry_prompt: str):
        """Log retry attempt with error feedback."""
        content = f"RETRY_ATTEMPT - Agent {self.agent_id} - Attempt {attempt}\n"
        content += f"Error Message: {error_message}\n"
        content += "Retry Prompt:\n"
        content += "=" * 50 + "\n"
        content += retry_prompt
        content += "\n" + "=" * 50
        self._write_to_log("VALIDATION", content)
    
    def log_chain_of_thought(self, reasoning: str, final_action: Dict[str, Any]):
        """Log Chain of Thought reasoning."""
        content = f"CHAIN_OF_THOUGHT - Agent {self.agent_id}\n"
        content += "Reasoning Process:\n"
        content += "=" * 50 + "\n"
        content += reasoning
        content += "\n" + "=" * 50 + "\n"
        content += f"Final Decision: {json.dumps(final_action, indent=2)}"
        self._write_to_log("REASONING", content)
    
    
    def log_final_decision(self, action: Dict[str, Any], execution_time: float):
        """Log final decision made by agent."""
        content = f"FINAL_DECISION - Agent {self.agent_id}\n"
        content += f"Action: {json.dumps(action, indent=2)}\n"
        content += f"Execution Time: {execution_time:.3f} seconds\n"
        content += "=" * 80
        self._write_to_log("DECISION", content)
    
    def log_error(self, error_type: str, error_message: str, context: str = ""):
        """Log errors with context."""
        content = f"ERROR - Agent {self.agent_id}\n"
        content += f"Error Type: {error_type}\n"
        content += f"Error Message: {error_message}\n"
        if context:
            content += f"Context: {context}"
        self._write_to_log("VALIDATION", content)
    
    def log_info(self, info_type: str, info_message: str, context: str = ""):
        """Log informational messages."""
        content = f"INFO - Agent {self.agent_id}\n"
        content += f"Info Type: {info_type}\n"
        content += f"Info Message: {info_message}\n"
        if context:
            content += f"Context: {context}"
        self._write_to_log("INFO", content)
    
    def log_debug(self, debug_type: str, debug_message: str, context: str = ""):
        """Log debug messages."""
        content = f"DEBUG - Agent {self.agent_id}\n"
        content += f"Debug Type: {debug_type}\n"
        content += f"Debug Message: {debug_message}\n"
        if context:
            content += f"Context: {context}"
        self._write_to_log("DEBUG", content)