from google import genai
import os

try: 
    client = genai.Client()
    print("Client initialized")
except Exception as e:
    print(f"Error initializing client: {e}")
    exit()
    
belief_graph_desc = "Your belief is that this card is not White about 80% of the time, and it is not a rank 1 about 80% of the time."
user_prompt = """
You are playing Hanabi-Full-CardKnowledge.

## GAME HISTORY


## HANABI RULES & MECHANICS

**GAME SETUP:**
- 5 colors (R,G,W,Y,B) × 5 ranks (1-5) = 25 unique cards
- 3 ones, 2 each of 2-4, 1 five per color

**OBJECTIVE:**
- Build fireworks: play 1→2→3→4→5 for each color
- Score: 25 points max (5 colors × 5 ranks)
- Playing a 5 earns +1 clue token

**FUNDAMENTAL RULE:**
- You CANNOT see your own cards
- You CAN see everyone else's cards
- CardKnowledge variant: explicit info about clues you receive

**CLUE SYSTEM:**
- 8 information tokens for hints, 3 life tokens
- Color clues: "These cards are [COLOR]" - affects ALL cards of that color
- Rank clues: "These cards are [RANK]" - affects ALL cards of that rank
- ABSOLUTE: Clued cards ALWAYS have the hinted property
- NEGATIVE: Un-clued cards NEVER have the hinted property
- NO ambiguity - clues are 100% truthful

**GAMEPLAY:**
- Wrong play costs 1 life, card goes to discard
- Game ends: 0 lives, empty deck, or complete fireworks
- You hold cards facing away, rely on teammate hints

## FORMAT REQUIREMENTS:
Respond with ONLY a valid JSON object in this exact format (ranks MUST use human numbers 1-5):
{"action_type": "PLAY|DISCARD|REVEAL_COLOR|REVEAL_RANK", "card_index": 0-4, "color": "R|Y|G|W|B", "rank": 1-5, "target_offset": 1-3}

**IMPORTANT**:
- For PLAY/DISCARD: use "card_index" (0-4), set color/rank/target_offset to null
- For REVEAL_COLOR: use "color" and "target_offset", set card_index/rank to null
- For REVEAL_RANK: use "rank" (1-5) and "target_offset", set card_index/color to null

## CURRENT GAME STATE:

## GAME OVERVIEW
- Information tokens: 7/8
- Life tokens: 3/3
- Deck size: 40
- Number of players: 2

## FIREWORKS PROGRESS
- R: 0/5
- Y: 0/5
- G: 0/5
- W: 0/5
- B: 0/5

## DISCARD PILE: empty

## OTHER PLAYERS' HANDS
Player 1: B5 R2 B2 B3 Y1 

## YOUR HAND
Card 1: Color: unknown, Rank: unknown
Card 2: Color: unknown, Rank: unknown
Card 3: Color: unknown, Rank: unknown
Card 4: Color: unknown, Rank: unknown
Card 5: Color: unknown, Rank: 1

## AVAILABLE ACTIONS
0: Discard card 0 (+1 info token)
1: Discard card 1 (+1 info token)
2: Discard card 2 (+1 info token)
3: Discard card 3 (+1 info token)
4: Discard card 4 (+1 info token)
5: Play card 0
6: Play card 1
7: Play card 2
8: Play card 3
9: Play card 4
10: Hint color R to player 1 (-1 info token)
11: Hint color Y to player 1 (-1 info token)
12: Hint color B to player 1 (-1 info token)
13: Hint rank 1 to player 1 (-1 info token)
14: Hint rank 2 to player 1 (-1 info token)
15: Hint rank 3 to player 1 (-1 info token)
16: Hint rank 5 to player 1 (-1 info token)


## DECISION REQUIRED:
Choose your action.

## CHAIN OF THOUGHT REQUIREMENT:

Before providing your final JSON action, you MUST provide detailed reasoning following this structure:

**REASONING:**
1. **SITUATION ANALYSIS**: What is the current game state? What are the key constraints?
2. **CARD KNOWLEDGE CHECK**: Do I know any 1s I can play? Do I know any other playable cards?
3. **TEAMMATE SCAN**: What 1s or immediately playable cards do I see in teammates' hands?
4. **LEGAL MOVES ANALYSIS**: What moves are available to me right now?
5. **STRATEGIC EVALUATION**: What would each legal move accomplish? What are the risks/benefits?
6. **PRIORITY ASSESSMENT**: 1s first, then other safe plays, then helpful hints, then safe discards
7. **FINAL DECISION**: Which action serves our team's goal best and why?

**FORMAT:**
First provide your detailed reasoning in the REASONING section above.
Then provide your action as a JSON object.

Example:
REASONING:
1. SITUATION ANALYSIS: We have 7 information tokens and 3 life tokens. Fireworks progress is slow with only Red at level 1.
2. LEGAL MOVES ANALYSIS: I can play 5 cards, discard 5 cards, or give 8 different hints.
3. STRATEGIC EVALUATION: Playing without certainty risks life tokens. Discarding wastes cards. Hinting helps teammate.
4. TEAMMATE CONSIDERATION: I see my teammate has a Red 2 that could be played next if they know about it.
5. PRIORITY ASSESSMENT: Information sharing is crucial right now to enable plays.
6. FINAL DECISION: I'll hint about the Red card to enable an immediate play.
"""

# `models.generate_content` is the correct call on the genai client
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_prompt + belief_graph_desc,
)


print(response.text)
