
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

