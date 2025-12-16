"""
Tests verifying production setup matching.

Ensures test prompts match production prompts exactly, and that
model configuration matches what was used in actual game runs.

Critical: Tests must use same prompts as production to catch real bugs.
"""

import pytest


def test_production_prompt_matches_test_prompt():
    """
    Critical: Test prompts must match production exactly.

    We must use same prompts as actual games, otherwise
    we might miss bugs like duplicate card issue.

    Steps:
    1. Load production prompt from prompt_manager.py
    2. Load test fixture prompt from tests/fixtures/production_prompts.txt
    3. Assert identical (or document differences)
    """
    # TODO:
    # from hanabi_learning_environment.agents.prompt_manager import PromptManager
    # pm = PromptManager()
    # production_rules = pm.get_rules_and_mechanics_block()
    #
    # with open("tests/fixtures/production_prompts.txt") as f:
    #     fixture_content = f.read()
    #     assert production_rules in fixture_content

    pytest.skip("Not implemented - need production prompt extraction")


def test_model_configuration_matches_laksh_setup():
    """
    Verify model matches what Laksh used in probabilistic runs.

    From instructions.md:
    - Should be: Gemini 2.0 Flash Lite
    - Currently is: Gemini 2.5 Pro (from gemini_agent.py:45)

    This test documents the discrepancy.
    """
    # TODO: Check agent model configuration
    # from hanabi_learning_environment.agents.gemini_agent import GeminiAgent
    # agent = GeminiAgent({})
    # agent.set_episode(1)
    # model_name = agent.model._model_name
    #
    # # This assertion will likely FAIL due to version mismatch
    # # Document in test output for investigation
    # assert "flash-lite" in model_name.lower() or "2.0" in model_name

    pytest.skip("Not implemented - need model config investigation")


def test_temperature_setting():
    """
    Verify temperature setting matches production.

    From exploration: temperature = 0.2 used in belief updates.

    This low temperature (closer to 0) ensures more deterministic output.
    """
    # TODO: Verify temperature from generation_config
    # Check both action decision and belief update calls

    pytest.skip("Not implemented - need model config investigation")


def test_belief_update_prompt_structure():
    """
    Verify belief update prompts include all required sections.

    Required sections:
    1. Base game context (Hanabi rules)
    2. Cooperative team context
    3. Game history
    4. Current game state
    5. Clue event details
    6. Belief update requirements
    7. Variant-specific instructions
    8. Response format
    """
    # TODO: Test prompt structure
    # from hanabi_learning_environment.agents.prompt_manager import PromptManager
    # pm = PromptManager()
    # event = {...}
    # prompt = pm.get_belief_update_prompt(event, ..., variant='probabilistic')
    #
    # assert "HANABI RULES" in prompt
    # assert "COOPERATIVE TEAM CONTEXT" in prompt
    # etc.

    pytest.skip("Not implemented - need production prompt extraction")


def test_cot_requirements_present():
    """
    Verify Chain of Thought requirements are in prompts.

    CoT should require structured reasoning before JSON output.
    """
    # TODO: Verify CoT requirements in prompts
    # from hanabi_learning_environment.agents.cot_reasoning import ChainOfThoughtReasoner
    # reasoner = ChainOfThoughtReasoner()
    # cot_prompt = reasoner.create_cot_prompt(...)
    #
    # assert "REASONING" in cot_prompt or "CHAIN OF THOUGHT" in cot_prompt

    pytest.skip("Not implemented - need CoT prompt investigation")
