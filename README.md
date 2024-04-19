# Agents Play Games Together

## Overview
This projects examines how two AI systems, with different personalities, can compete against eachother.  It is still a work in progress as I initially wanted to implement it in Catan, which was probably too complicated.  Right now, I have a working battleship environment, but it will need to be refined before people can play/watch the AIs play the game.

## Running the environment
To test the project, just run "battleship_game.py" in a python environment that includes vllm and huggingface.  You might need extra permissions to run the Llama-3 model.

## Evaluation
I tested the project with both Llama3-8B and Wizard-2-7B.  The Wizard model performed significantly better when it came to chatting, as the Llama3-8B model's chat messages were not related to the game and were also pretty innappropriate.  I want to experiment with applying a larger model, either Wizard-2-8x22B or Gemini.

## Future directions
For my final project, I plan to implement Catan and find a way to run the model on a more advanced network.
