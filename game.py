import numpy as np
from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type
from abc import ABC, abstractmethod
import string

# R^n * R^(n*m)
GameResult = Tuple[np.ndarray, np.ndarray]


class GameBase(ABC):
    @abstractmethod
    def __init__(self):
        self.n = 0
        self.m = 0

    # R^(n*m) -> R^n
    def value(self, *args: np.ndarray) -> np.ndarray:
        return self(*args)[0]
        
    # R^(n*m) -> R* ^ (n*m)
    def gradient(self, *args: np.ndarray) -> np.ndarray:
        return self(*args)[1]

    # R^(n*m) -> R^n * R* ^ (n*m)
    def __call__(self, *args: np.ndarray) -> GameResult:
        return self.value(*args), self.gradient(*args)


class MultilinearGame(GameBase):
    def __init__(self, weight: np.ndarray):
        self.weight = weight
        self.m = weight.shape[1]
        self.n = weight.ndim - 1
        assert self.n == weight.shape[0]
        chars = string.ascii_lowercase
        self.einsum_str = '{}{},{}->{}'.format(
            chars[self.n], chars[:self.n], ','.join(chars[:self.n]), chars[self.n]
        )
        self.single_einsum_str = [
                '{},{}->{}'.format(
                chars[:self.n], ','.join(chars[:self.n]), chars[i]
            )
            for i in range(self.n)
        ]
        
    def value(self, *args: np.ndarray) -> np.ndarray:
        return np.einsum(self.einsum_str, self.weight, *args)

    def gradient(self, *args: np.ndarray) -> np.ndarray:
        l : List[np.ndarray] = []
        for i in range(self.n):
            g_inputs = list(args)
            g_inputs[i] = np.ones_like(g_inputs[i])
            l.append(np.einsum(self.single_einsum_str[i], self.weight[i], *g_inputs))
        
        return np.array(l)