# Gym-Wordle

An OpenAI gym compatible environment for training agents to play Wordle.

![ezgif com-gif-maker](https://user-images.githubusercontent.com/8514041/152437216-d78e85f6-8049-4cb9-ae61-3c015a8a0e4f.gif)

## Installation

```
$ pip install gym_wordle
```

## Usage

Here is an example script that allows you to play Wordle as a human.

```Python
import gym
import gym_wordle

from gym_wordle.utils import to_array, to_english

env = gym.make('Wordle-v0')

env.reset()

done = False

while not done:
    env.render()
    valid = False

    while not valid:
        guess = input('Guess: ').lower()
        action = to_array(guess)

        if env.action_space.contains(action):
            valid = True

    state, reward, done, info = env.step(action)

env.render()

print(f"The word was {to_english(env.solution).upper()}")
```

The above script is more or less equivalent to the function `play()` found in
`gym_wordle.utils`.

## Examples
