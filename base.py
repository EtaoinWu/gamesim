import numpy as np
from abc import ABC, abstractmethod

from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type


def holder_conjugate(x):
    if x == np.inf:
        return 1
    if x == 1:
        return np.inf
    return 1 / (1 - 1 / x)


class Proj(ABC):
    @abstractmethod
    def __call__(self, v: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def best_pure(self, v: np.ndarray) -> np.ndarray:
        pass


class ProjNonnegative(Proj):
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0)

    def best_pure(self, v: np.ndarray) -> np.ndarray:
        return (v > 0).astype(float)


class ProjCube(Proj):
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(v, 0), 1)

    def best_pure(self, v: np.ndarray) -> np.ndarray:
        return (v > 0).astype(float)


class ProjUnbounded(Proj):
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        return v

    def best_pure(self, v: np.ndarray) -> np.ndarray:
        return np.zeros_like(v)


class ProjDimension(Proj):
    def __init__(self, base: np.ndarray):
        self.base = base

    def __call__(self, v: np.ndarray) -> np.ndarray:
        if v.ndim < self.base.ndim:
            return v
        if v.ndim > self.base.ndim:
            return np.array([self(v[i]) for i in range(v.shape[0])])
        return np.maximum(v, self.base)

    def best_pure(self, v: np.ndarray) -> np.ndarray:
        return np.zeros_like(v)


# https://arxiv.org/abs/1309.1541
class ProjSim(Proj):
    @staticmethod
    def __call__(y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            d = y.shape[0]
            u = np.sort(y)[::-1]
            v = np.cumsum(u)
            ρ = max([j for j in range(d) if u[j] + (1 - v[j]) / (j + 1) > 0])
            λ = (1 - v[ρ]) / (ρ + 1)
            return np.maximum(y + λ, 0)
        else:
            return np.array([ProjSim.__call__(y[i]) for i in range(y.shape[0])])

    @staticmethod
    def best_pure(y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            i = y.argmax()
            return (np.arange(y.shape[0]) == i).astype(float)
        else:
            return np.array([ProjSim.best_pure(y[i]) for i in range(y.shape[0])])

class ProjBall(Proj):
    def __init__(self, ord: Any):
        self.ord = ord
        self.dual_ord = holder_conjugate(ord)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            if self.ord == 2:
                ρ = np.linalg.norm(y, ord=self.ord)
                return min(1, 1. / ρ) * y
            elif self.ord == 1:
                # projection onto L1 ball
                x = np.abs(y)
                if np.sum(x) <= 1:
                    return y
                z = ProjSim.__call__(x)
                return np.sign(y) * z
            elif self.ord == np.inf:
                # projection onto Linf ball
                return y.clip(-1, 1)
        else:
            return np.array([self(y[i]) for i in range(y.shape[0])])

    def best_pure(self, y: np.ndarray) -> np.ndarray:
        if y.ndim == 1:
            if self.ord == 1:
                return np.sign(y) * (np.arange(y.shape[0]) == np.argmax(np.abs(y))).astype(float)
            x = np.abs(y) ** (self.dual_ord - 1) * np.sign(y)
            ρ = np.linalg.norm(x, ord=self.ord)
            return 1. / ρ * x
        else:
            return np.array([self.best_pure(y[i]) for i in range(y.shape[0])])


proj_trivial = ProjUnbounded()
proj_nn = ProjNonnegative()
proj_cube = ProjCube()
proj_sim = ProjSim()
proj_linfball = ProjBall(np.inf)
proj_l2ball = ProjBall(2)
proj_l1ball = ProjBall(1)
