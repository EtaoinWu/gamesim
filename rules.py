import numpy as np
from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type
from abc import ABC, abstractmethod
from base import Proj

UpdateResult = Tuple[np.ndarray, Any]


class UpdateRule(ABC):
    @abstractmethod
    def init_internal(self, n: int, m: int):
        return None

    @abstractmethod
    def __call__(self, play: np.ndarray, internal: Any, util: np.ndarray, grad: np.ndarray) -> UpdateResult:
        pass


class GradientDescent(UpdateRule):
    def __init__(self, eta: float, proj: Proj):
        self.eta = eta
        self.proj = proj

    def init_internal(self, n, m):
        return None

    def __call__(self, play: np.ndarray, _, util: np.ndarray, grad: np.ndarray) -> UpdateResult:
        return self.proj(play - self.eta * grad), None


class OptimisticGradient(UpdateRule):
    def __init__(self, eta: float, proj: Proj):
        self.eta = eta
        self.proj = proj

    def init_internal(self, n, m) -> np.ndarray:
        return np.array([self.proj(np.zeros(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray) -> UpdateResult:
        wholestep = internal
        # halfstep = play
        next_wholestep = self.proj(wholestep - self.eta * grad)
        next_halfstep = self.proj(next_wholestep - self.eta * grad)
        return next_halfstep, next_wholestep


class ExtraGradient(UpdateRule):
    def __init__(self, eta: float, proj: Proj):
        self.eta = eta
        self.proj = proj

    InternalType = Tuple[bool, np.ndarray]

    def init_internal(self, n, m) -> InternalType:
        return True, np.array([self.proj(np.zeros(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal : InternalType, util: np.ndarray, grad: np.ndarray) -> UpdateResult:
        if internal[0]:
            return self.proj(play - self.eta * grad), (not internal[0], play)
        else:
            return self.proj(internal[1] - self.eta * grad), (not internal[0], play)
