import numpy as np
from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type

class Proj:
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        pass


class ProjNonnegative(Proj):
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        return np.maximum(v, 0)


class ProjCube(Proj):
    @staticmethod
    def __call__(v: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(v, 0), 1)

# https://arxiv.org/abs/1309.1541
class ProjSim(Proj):
    @staticmethod
    def __call__(y: np.ndarray) -> np.ndarray:
        d = y.shape[0]
        u = np.sort(y)[::-1]
        v = np.cumsum(u)
        ρ = max([j for j in range(d) if u[j] + (1 - v[j]) / (j + 1) > 0])
        λ = (1 - v[ρ]) / (ρ + 1)
        return np.maximum(y + λ, 0)


proj_nn = ProjNonnegative()
proj_cube = ProjCube()
proj_sim = ProjSim()
