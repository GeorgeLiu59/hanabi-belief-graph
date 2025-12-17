"""
Hanabi Belief Graph Unit Tests

Test suite for validating belief graph probability calculations in multi-agent
Hanabi research. These tests catch bugs where impossible cards receive non-zero
probabilities.

Critical Bugs Being Tested:
- Bug #1: Uniform probability reset (Turn 28 bug)
- Bug #2: Duplicate card misunderstanding

Test Organization:
- test_belief_initialization.py: Turn 28 bug reproduction
- test_belief_updates.py: Multi-turn progression tests
- test_input_formats.py: Natural language vs JSON formatting
- test_prompt_consistency.py: Production setup verification
"""
