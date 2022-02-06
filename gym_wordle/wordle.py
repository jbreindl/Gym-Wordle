import gym
import numpy as np
import numpy.typing as npt
from sty import fg, bg, ef, rs

from collections import defaultdict
from gym_wordle.utils import to_english, to_array, get_words
from typing import Optional 


class WordList(gym.spaces.MultiDiscrete):
    """Super class for defining a space of valid words according to a specified
    list.

    At present, the space is a subclass of gym.spaces.MultiDiscrete, all
    elements of the space are one-dimensional, length five integer arrays
    containing integers from 0,...,26 (inclusive).

    This class overrides the contains and sample methods from the MultiDiscrete
    parent, because the additional qualification for elements in this space is
    that they be entries in the list of words provided.
    """

    def __init__(self, words: npt.NDArray[np.int64], **kwargs):
        """
        Args:
            words: Collection of words in array form with shape (_, 5), where
              each word is a row of the array. Each array element is an integer
              between 0,...,26 (inclusive).
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        super().__init__([26] * 5, **kwargs)
        self.words = words

    def contains(self, x: npt.NDArray[np.int64]) -> bool:
        """Checks whether a given word is an element of the space.

        Args:
            x: Word in array form. It is assumed that x has shape (5,)
              and is composed of integers between 0,...,26 (inclusive). 

        Returns:
            True if and only if x corresponds to a row in the list of words
            comprising the space.
        """
        return (x == self.words).all(axis=1).any()

    def sample(self) -> npt.NDArray[np.int64]:
        """Samples a word from the space.

        Returns:
            An entry from the list of words provided to the space in array
            form.
        """
        # TODO: use the builtin random capabilities from the super class
        index = self.np_random.randint(self.words.shape[0])
        return self.words[index]


class SolutionList(WordList):
    """Space for *solution* words to the Wordle environment.

    In the game Wordle, there are two different collections of words:

        * "guesses", which the game accepts as valid words to use to guess the
          answer.
        * "solutions", which the game uses to choose solutions from.

    Of course, the set of solutions is a strict subset of the set of guesses.

    Reference: https://fivethirtyeight.com/features/when-the-riddler-met-wordle/

    This class represents the set of solution words.
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        words = get_words('solution')
        super().__init__(words, **kwargs)


class GuessList(WordList):
    """Space for *solution* words to the Wordle environment.

    In the game Wordle, there are two different collections of words:

        * "guesses", which the game accepts as valid words to use to guess the
          answer.
        * "solutions", which the game uses to choose solutions from.

    Of course, the set of solutions is a strict subset of the set of guesses.

    Reference: https://fivethirtyeight.com/features/when-the-riddler-met-wordle/

    This class represents the set of guess words.
    """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: See documentation for gym.spaces.MultiDiscrete
        """
        words = get_words('guess')
        super().__init__(words, **kwargs)


class WordleEnv(gym.Env):

    metadata = {'render.modes': ['human']}

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

    def render(self, mode='human', **kwargs):
        assert mode == 'human'
        
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

        # Total reward equals -(number of incorrect guesses)
        reward = 0. if correct else -1.

        return self.state, reward, done, {}

    def seed(self, seed: Optional[int]= None): 
        """
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
