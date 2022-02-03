from wordlebot.wordle import WordleEnv
from wordlebot.utils import to_array, to_english

env = WordleEnv()
env.reset()
env.solution = to_array('sheet')
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

