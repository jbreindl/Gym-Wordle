import numpy as np
import numpy.typing as npt

from pathlib import Path


_chars = ' abcdefghijklmnopqrstuvwxyz'
_char_d = {c: i for i, c in enumerate(_chars)}


def to_english(array: npt.NDArray[np.int64]) -> str:
    """
    Converts a numpy integer array into a corresponding English string.

    Params
    ------
    array: 
    """
    return ''.join(_chars[i] for i in array)


def to_array(word: str) -> npt.NDArray[np.int64]:
    """
    Converts a string of characters into a corresponding numpy array
    """
    return np.array([_char_d[c] for c in word])


def get_words(category: str, build: bool=False) -> npt.NDArray[np.int64]:
    """
    """
    assert category in {'guess', 'solution'}
    
    arr_path = Path(__file__).parent / f'dictionary/{category}_list.npy'
    if build:
       list_path = Path(__file__).parent / f'dictionary/{category}_list.csv'

       with open(list_path, 'r') as f:
           words = np.array([to_array(line.strip()) for line in f])
           np.save(arr_path, words)

    return np.load(arr_path)
        
