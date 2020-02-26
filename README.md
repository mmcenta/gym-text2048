# gym-text2048

2048 as an OpenAI Gym environment with a simple text display. Currently, the tasks implemented are:

## Text2048

This is the standard game. The rewards are the sum of the merged tiles in the step (which is the base for the score in the original game). An episode ends only when no more moves are possible. As a result, it is possible to achieve tiles bigger than 2048. While the state space is limited, with 2<sup>31</sup> being the maximum value for a tile, the maximum tile reached by the best AI solvers is usually 32,768 or 2<sup>15</sup>. For this reason, I believe this implementation to be infinite for practical purposes.

# Installation

```bash
cd gym-text2048
pip install -e .
```
