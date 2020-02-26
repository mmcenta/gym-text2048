# gym-text2048

2048 as an OpenAI Gym environment with a simple text display. Currently, the tasks implemented are:

## Text2048

This is the standard game. The rewards are the sum of the merged tiles in the step (which is the base for the score in the original game). An episode ends only when no more moves are possible. As a result, it is possible to achieve tiles bigger than 2048 - we limit the state space by noting that <img src="https://render.githubusercontent.com/render/math?math=2^{n^2 + 1}"> is an upper limit for the value of tiles in a <img src="https://render.githubusercontent.com/render/math?math=n"> by <img src="https://render.githubusercontent.com/render/math?math=n"> board.

# Installation

```bash
cd gym-text2048
pip install -e .
```
