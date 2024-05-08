# Agents Play Games Together
**Ian Whitehouse**
[Associated Presentation](presentation.pptx)

## Overview
Inspired by “Generative Agents: Interactive Simulacra of Human Behavior,” this project aims to simulate multiple language models playing a game of battleshps.  Initially, I planned to expand this project to other games, such as the popular multiplayer board game Catan, however, the language models struggled on Battleship, so I thought that Catan would likely be too complicated for them to perform well.  Therefore, for my final project, I focused on improving how the agents played Battleship.

On my initial draft of this project, two agents could play against eachother or an agent could play against a user.  Agents were given personality traits, which were fed into their prompts, both when they chatted and when they played the game.  Each round, the agents made a move in the game and sent one chat message.

Agents saw the board as a square, filled with either their ship positions or their hits/misses.  Their ships were randomly placed and they were shown where those ships were placed at the beginning of the game.  Everytime they made a move, they were reminded of their past moves.  Below is an example prompt that an agent was given when it made a move.  Previous moves were shown above this prompt.

```
This board shows where you have attacked the user's ships (your hits appear as X and your misses appear as O):
  0 |   1 |   2 |   3 |   4 |   5 |   6
  7 |   8 |   9 |  10 |  11 |  12 |  13
 14 |  15 |  16 |  17 |  18 |  19 |  20
 21 |  22 |  23 |  24 |  25 |  26 |  27
 28 |  29 |  30 |  31 |  32 |  33 |  34
 35 |  36 |  37 |  38 |  39 |  40 |  41
 42 |  43 |  44 |  45 |  46 |  47 |  48
It is generally a good idea to attack near your previous successful hits, as they show that there is a ship in that area.  However, you should not just attack every spot in order because you must use logic and reasoning to decide which area to attack.  Where would you like to attack? Choose a number between 0 and 48 that IS NOT in this list: .  You should respond with a single sentence of your reasoning BEFORE giving the number of the area that you want to attack.<|start_header_id|>assistant<|end_header_id|>
```

After they made a move, they sent one chat message, which, as I stated before, was based on their personality.  Below is an example of chat messages that an angry agent sent to a calm agent (labeled as the user).

```
system: Agent attacked 0 and missed
agent: ARE YOU KIDDING ME?! I JUST WASTED MY FIRST ATTACK ON A BLANK SPACE?! YOU THINK YOU'RE SO SMART, DON'T YOU?! WELL, I'VE GOT NEWS FOR YOU, PAL! I'M NOT GOING DOWN WITHOUT A FIGHT! NEXT, I'LL ATTACK... HMMM... HOW ABOUT SPACE 16?!
system: User attacked 0 and missed
user: No need to get worked up, my friend. It's just a game, and we're just getting started. You're right, though - you did miss on space 0. But don't worry, you've got plenty of chances left. Now, let's see about that attack on space 16...
system: Agent attacked 1 and missed
agent: WHAT THE...?! I MISSED AGAIN?! YOU'RE JUST SITTING THERE, SMILING, THINKING YOU'VE GOT THE UPPER HAND, AREN'T YOU?! WELL, LET ME TELL YOU SOMETHING, PAL! I'M NOT GOING TO GIVE UP! I'LL KEEP COMING AT YOU UNTIL I SINK ALL YOUR PATHETIC LITTLE SHIPS! NEXT, I'LL ATTACK... UGH... SPACE 23!

```

## Improving the model
For the final part of this project, I focused on improving how the language models played the game.  This included asking the models' for their reasoning before they made their move, suggesting that they should attack nearby their previous successful hits, and reprompting them with additional instruction if they attempted to make an invalid move.

However, what worked the best was switching to an instruction-tuned model and then to a larger model.  The initial project used the *WizardLM-2-7B* model, so the first change I made was redo prompting to use a chat template and begin using the *Meta-Llama-3-8B-Instruct* model.  This model was better, but I achieved the best results with a quantized *Meta-Llama-3-70B-Instruct* model.

Ultimately, most of my time on this project was spent adding the *Meta-Llama-3-70B-Instruct* model using vLLM.  Even the quantized version I used took nearly 40GB of VRAM and 170GB of system RAM, and, while running it, vLLM was incredibly unstable.  I think that there might be something wrong or incorrectly configured with vLLM or the quantized model that made this an issue.

## Running the environment
To test the project, just run "battleship_game_llama_instruct.py" in a python environment that includes vLLM and HuggingFace.  You might need extra permissions to run the Llama-3 model and you might need Ray to allow vLLM to use multiple graphics cards.

## Evaluation
It was hard to do objective tests on the model because of vLLM crashes, which meant that I would have to restart the program.  However, the model does pretty poorly playing Battleships.  According to the DataGenetics.com blog, the optimal battleship player will attack 30% of the board, while, when games do finish, the language models have attacked 90%, suggesting that they may not be better than random chance.

## Future directions
The first step to finishing this project would be to apply a larger model or apply a non-quantized *Meta-Llama-3-70B-Instruct* model.  In hindsight, it was a mistake to spend so long attempting to the run *Meta-Llama-3-70B-Instruct* model locally before piviting to an API-based service.  However, there is a future, more powerful model called *Meta-Llama-3-400B-Instruct* which I look forward to trying on this project.

Additionally, I want to integrate the chat function and the agent's personality more closely into the gameplay.  Unfortunately, Battleships may have been a bad choice of game as it isn't very social.  I initially wanted to use the models to play Catan, which is a more social game that, in my experience, is largely driven by people's personalities.

Finally, after seeing the language models master Battleship, I would like to turn this project into a general framework that could be used to play any game, especially more complicated ones.
