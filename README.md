# gym-text2048

2048 as an OpenAI Gym environment with a simple text display.

## Environments

This package implements several tasks based on the game 2048. They share the following similarities:

* The **action space** is an instance of `Discrete` with 4 elements, with one for each direction (0 for up, 1 for right, 2 for down and 3 for left);
* The **observation space** is an instance of `MultiDiscrete`, in which each tile of the board is discrete space (the number of elements varies with the board size and whether the task is capped or not). The observations are the logarithmic of base two of the tile values - for example, a tile with value 2048 is represented by an 11 in a observation.

The reward and the end condition depend on the task. Currently, the implemented tasks are:

### Text2048

The Text2048 task models the standard game. The rewards are the sum of the merged tiles in the step (which is the base for the score in the original game). An episode ends only when no more moves are possible. As a result, it is possible to achieve tiles bigger than 2048 - we limit the state space by noting that 2<sup>n<sup>2</sup> + 1</sup> is an upper limit for the value of tiles in a n by n board.

### Text2048WithHeuristic

Similar to Text2048, but instead of the original rewards, it instead uses a heuristic reward. See the Heuristic Reward subsection for more information.

### Text2048Capped

Similar to Text2048, but it stops the game once the maximum tile of the board has reached a value (2048 by default). As a result, the observation space for each tile is limited by the value of the maximum tile.

### Text2048CappedWithHeuristic

This enviroment implements the changes of Text2048WithHeuristic and Text2048Capped into a single task.

## Examples

This repository includes the following examples under the directory `/examples`:

* `play.py` which allows the user to play the game using the WASD keys;
* `random.py` which implements a random agent.

## Installation

Installation can be done locally using `pip`, just download the repository's files and execute the following commands:

```bash
cd gym-text2048
pip install -e .
```
