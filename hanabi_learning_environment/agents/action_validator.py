"""Action validation module for Hanabi agents."""

import json
import random
from typing import Dict, Any, List, Optional, Tuple


class ActionValidator:
    """Handles action parsing, validation, and legal move checking."""
    
    def parse_llm_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response and extract the action."""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                action = json.loads(json_str)
                
                # Clean up the action to fix common LLM format issues
                action = self._clean_action_format(action)
                
                return action
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return None
        return None
    
    def _clean_action_format(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up common LLM format issues in actions."""
        cleaned = action.copy()
        
        # Fix card_index issues
        if 'card_index' in cleaned:
            if cleaned['card_index'] == -1 or cleaned['card_index'] is None:
                # For hints, card_index should be None
                if cleaned['action_type'] in ['REVEAL_COLOR', 'REVEAL_RANK']:
                    cleaned['card_index'] = None
                # For play/discard, card_index should be 0-4
                elif cleaned['action_type'] in ['PLAY', 'DISCARD']:
                    cleaned['card_index'] = 0  # Default to first card
        
        # Ensure required fields are present and correct
        if cleaned['action_type'] == 'PLAY':
            if 'card_index' not in cleaned or cleaned['card_index'] is None:
                cleaned['card_index'] = 0
            cleaned['color'] = None
            cleaned['rank'] = None
            cleaned['target_offset'] = None
            
        elif cleaned['action_type'] == 'DISCARD':
            if 'card_index' not in cleaned or cleaned['card_index'] is None:
                cleaned['card_index'] = 0
            cleaned['color'] = None
            cleaned['rank'] = None
            cleaned['target_offset'] = None
            
        elif cleaned['action_type'] == 'REVEAL_COLOR':
            # Don't set defaults - let validation handle missing values
            cleaned['card_index'] = None
            
        elif cleaned['action_type'] == 'REVEAL_RANK':
            # Don't set defaults - let validation handle missing values
            cleaned['card_index'] = None
            cleaned['color'] = None
        
        return cleaned
    
    def is_action_legal(self, action: Dict[str, Any], legal_moves: List[Dict[str, Any]]) -> bool:
        """Check if an action is legal by comparing against legal moves."""
        for legal_move in legal_moves:
            if (action.get('action_type') == legal_move.get('action_type') and
                action.get('card_index') == legal_move.get('card_index') and
                action.get('color') == legal_move.get('color') and
                action.get('rank') == legal_move.get('rank') and
                action.get('target_offset') == legal_move.get('target_offset')):
                return True
        return False
    
    def get_illegal_reason(self, action: Dict[str, Any], legal_moves: List[Dict[str, Any]]) -> str:
        """Generate a clear explanation of why an action is illegal."""
        action_type = action.get('action_type')
        
        # Check if action type exists in legal moves
        legal_action_types = [move.get('action_type') for move in legal_moves]
        if action_type not in legal_action_types:
            return f"Action type '{action_type}' is not available. Legal types: {list(set(legal_action_types))}"
        
        # Check specific parameters for this action type
        if action_type == 'REVEAL_RANK':
            rank = action.get('rank')
            legal_ranks = [move.get('rank') for move in legal_moves if move.get('action_type') == 'REVEAL_RANK']
            if rank not in legal_ranks:
                return f"REVEAL_RANK with rank {rank} is not legal. Available ranks: {legal_ranks}"
                
        elif action_type == 'REVEAL_COLOR':
            color = action.get('color')
            legal_colors = [move.get('color') for move in legal_moves if move.get('action_type') == 'REVEAL_COLOR']
            if color not in legal_colors:
                return f"REVEAL_COLOR with color {color} is not legal. Available colors: {legal_colors}"
                
        elif action_type in ['PLAY', 'DISCARD']:
            card_index = action.get('card_index')
            legal_indices = [move.get('card_index') for move in legal_moves if move.get('action_type') == action_type]
            if card_index not in legal_indices:
                return f"{action_type} with card_index {card_index} is not legal. Available indices: {legal_indices}"
        
        return f"Action {action} does not match any legal move exactly."
    
    
    def validate_action_completeness(self, action: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate that an action has all required fields."""
        if not action:
            return False, "Action is None or empty"
        
        action_type = action.get('action_type')
        if not action_type:
            return False, "Missing action_type field"
        
        if action_type == 'PLAY':
            if action.get('card_index') is None:
                return False, "PLAY action missing card_index"
                
        elif action_type == 'DISCARD':
            if action.get('card_index') is None:
                return False, "DISCARD action missing card_index"
                
        elif action_type == 'REVEAL_COLOR':
            if action.get('color') is None:
                return False, "REVEAL_COLOR action missing color"
            if action.get('target_offset') is None:
                return False, "REVEAL_COLOR action missing target_offset"
                
        elif action_type == 'REVEAL_RANK':
            if action.get('rank') is None:
                return False, "REVEAL_RANK action missing rank"
            if action.get('target_offset') is None:
                return False, "REVEAL_RANK action missing target_offset"
        else:
            return False, f"Unknown action_type: {action_type}"
        
        return True, "Action is complete"