Phi 3.5 tests

- size vs toks/sec
- toks/sec vs thinking time
- acc vs thinking time

Plan:

- Finetune for problem formatting & syntax understanding

- Generate a few solutions and then treat these partial solutions as the state for gflownets
- Determine inference-time scaling properties of phi 3.5 / 4
- Test-time training of "good" and "bad" trains of thought
- Use pivotal tokens to construct a graph???

### Important ideas / methods to try:

- Get USAMO and other olympiad questions

- KTO
- DPO w/ Pivotal tokens
- Time-based prompting (insert "you have X time left" as time goes on)
- More prompting methods:

  - "inquisitive, curious, scientific, analytical"
  - "genius, expert, brilliant"
  - "efficient, effective"
  - "thorough, methodical"

- Figure out how to create state to do gflownets
  - define a quantum "step" of a question using datasets of solutions? Use these as the "step" vectors??
    - To validate, find similar hidden state "directions" or activations over several model solutions and see
      if the outputs are also similar in direction. See if I can cluster these "directions" and get general
      steps for math problem solving.
    - A prerequisite here is being able to effectively and efficiently extract & cluster "activations" or "embeddings"
      or whatever signal we're looking for from all these LLMs
  - If I can do this, then we're in a good spot
- Model conversations
- Compare performance of different compositions of models

## Plan:

### Training:

grpo and pto can be complimentary.
the completion length of grpo is one of its weaknesses.
the scalability / data efficiency of pts is one of its weaknesses.

Base Models: - Deepseek distilled - Distill your own (?)
Algos: - KTO - Good / Bad formatting - DPO - Pivotal tokens strategy of Phi 4 - SFT - Numina TIR dataset

Data Generation:

- Study up on prompting methods
  - Deepseek v3
  - Deepseek R1
    - Prompt? How to evaluate prompt effectiveness?

Validation: - Curate a set of AIME-level math questions - Switch numbers around perhaps? - Create my own of varying difficulty olympiad-level questions

Inference: - Hyperparameters - Model mixtures - % splits between models - \*\*\* model conversations??

Something I need to keep in mind is to set this project upon manageable abstract footing. I don't want to end up with a situation where the code is difficult to work with and slows me down. What do I see as the usable modules of this?

- Training: being able to run training runs with my pre-determined variables and determined datasets
- Data generation: being able to generate data and format the output so that I can post-process it with my variables
- Inference: being able to download the model and do iterations of model inference with variables

Data generation pipeline
Model(Prompt) -> Problem

Prompt:

    purpose:
    - output format IS [b]oxed{}
    - let model try to solve problems
    - output is number between 1 and 1000
    - difficulty level AIME
    - model can do TIR (dataset has TIR formatting)

    unwritten rules of prompt engineering?

    After: verify structure, determine good / bad outputs for KTO

Pivotal token method:

for text in question_texts:
dataset = [Model(text)[:K] + answer_prompt for K in len(text)]
Model.generate(dataset, stop after boxed or max len) -> [answer or null for K in len(text)]
