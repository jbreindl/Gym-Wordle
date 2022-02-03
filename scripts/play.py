import gym
import gym_wordle

from gym_wordle.utils import to_array, to_english

env = gym.make('Wordle-v0')

env.reset()
done = False

while not done:
    env.render()
    good_guess = False

    while not good_guess:

        guess = input('Guess: ').lower()
        action = to_array(guess)

        if env.action_space.contains(action):
            good_guess = True

    state, reward, done, info = env.step(action)

env.render()

print(f"The word was {to_english(env.solution).upper()}")

