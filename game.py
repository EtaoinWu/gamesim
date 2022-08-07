import numpy as np
from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type
from abc import ABC, abstractmethod
import string
import torch
from torch import nn

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
        wt = torch.tensor(weight)
        self.weight = wt
        self.m = wt.shape[1]
        self.n = wt.ndim - 1
        assert self.n == wt.shape[0]
        chars = string.ascii_lowercase
        self.einsum_str = '{}{},{}->{}'.format(
            chars[self.n], chars[:self.n], ','.join(chars[:self.n]), chars[self.n]
        )
        self.single_einsum_str = '{},{}'.format(
            chars[:self.n], ','.join(chars[:self.n])
        )
        
    def value(self, *args: np.ndarray) -> np.ndarray:
        inputs = list(map(lambda x: torch.tensor(x), args))
        outs = torch.einsum(self.einsum_str, self.weight, *inputs)
        return np.array(outs.detach().numpy())

    def gradient(self, *args: np.ndarray) -> np.ndarray:
        l : List[np.ndarray] = []
        for i in range(self.n):
            g_inputs = list(map(lambda x: torch.tensor(x, requires_grad=True), args))
            torch.einsum(self.single_einsum_str, self.weight[i], *g_inputs).backward()
            l.append(np.array(g_inputs[i].grad.detach().numpy()))
        
        return np.array(l)