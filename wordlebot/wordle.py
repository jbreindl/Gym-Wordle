import gym
import numpy as np
from sty import fg, bg, ef, rs
from collections import defaultdict

from wordlebot.utils import to_english, to_array, get_words


class WordList(gym.spaces.MultiDiscrete):

    def __init__(self, words, **kwargs):
        super().__init__([26] * 5, **kwargs)
        self.words = words

    def contains(self, x):
        return (x == self.words).all(axis=1).any()

    def sample(self) -> np.ndarray:
        index = self.np_random.randint(self.words.shape[0])
        return self.words[index]


class SolutionList(WordList):

    def __init__(self, **kwargs):
        super().__init__(get_words('solution'), **kwargs)


class GuessList(WordList):

    def __init__(self, **kwargs):
        super().__init__(get_words('guess'), **kwargs)


class WordleEnv(gym.Env):

    def __init__(self):
        super().__init__()

        self.seed()
        self.action_space = GuessList()
        self.solution_space = SolutionList()

        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Dict({
                'word': GuessList(),
                'flags': gym.spaces.MultiDiscrete([4] * 5)
            }) for _ in range(6))
        )

    def reset(self):
        self.round = 0
        self.solution = self.solution_space.sample()

        self.state = tuple({
            'word': np.zeros(5, dtype=np.int64), 
            'flags': np.zeros(5, dtype=np.int64)
        } for _ in range(6))

        return self.state

    def render(self):
        
        for row in self.state:
            row_fmt = []
            for c, f in zip(to_english(row['word']), row['flags']):
                c = c.upper()
                if f == 0:
                    row_fmt.append(c)
                elif f == 1:
                    row_fmt.append(bg.green + c + bg.rs)
                elif f == 2:
                    row_fmt.append(bg.yellow + c + bg.rs)
                else:
                    row_fmt.append(c)
            print(''.join(row_fmt))
                
    def close(self):
        pass

    def step(self, action):

        assert self.action_space.contains(action), 'Invalid word!'

        self.state[self.round]['word'] = action
        char_counter = defaultdict(int)
        for i, c in enumerate(action):
            # process flags for each character
            char_counter[c] += 1
            if c == self.solution[i]:
                # this is the right character in the right position
                self.state[self.round]['flags'][i] = 1
            elif char_counter[c] <= (c == self.solution).sum():
                # this is the right character, but it's not in the right
                # position. 
                self.state[self.round]['flags'][i] = 2
            else:
                self.state[self.round]['flags'][i] = 3

        self.round += 1
        correct = (action == self.solution).all()
        game_over = (self.round == 6)

        done = correct or game_over

        if correct:
            reward = 1.
        elif done:
            reward = -1.
        else:
            reward = 0.

        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
