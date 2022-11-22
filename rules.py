import numpy as np
from typing import TypeVar, Generic, List, Tuple, Dict, Callable, Union, Optional, Any, Type, TypeAlias
from abc import ABC, abstractmethod
from base import Proj
from itertools import islice, cycle
import scipy
import scipy.optimize

UpdateResult = Tuple[np.ndarray, Any]
LearningRate = Union[Callable[[int], float], float]


def learning_rate(lr: LearningRate, T: int) -> float:
    if callable(lr):
        return lr(T)
    return lr


class UpdateRule(ABC):
    @abstractmethod
    def init_internal(self, n: int, m: int):
        return None

    # play: array(n,m)
    # internal: InternalType
    # util: array(n)
    # grad: array(n,m)
    # T: int
    # -> (array(n,m),InternalType)
    @abstractmethod
    def __call__(self, play: np.ndarray, internal: Any, util: np.ndarray, grad: np.ndarray, T: int) -> UpdateResult:
        pass

class Alternating(UpdateRule):
    InternalType = tuple[int, list[Any]]

    def __init__(self, sub: Union[list[UpdateRule], UpdateRule]):
        if isinstance(sub, list):
            self.sub = sub
        else:
            self.sub = [sub]

    def init_internal(self, n: int, m: int):
        return 0, [x.init_internal(1, m) for x in islice(cycle(self.sub), n)]

    def __call__(self, play: np.ndarray, internal: InternalType, util: np.ndarray, grad: np.ndarray,
                 T: int) -> Tuple[np.ndarray, InternalType]:
        i, sub_internals = internal
        n = len(sub_internals)
        actual_time = T // n + 1
        active_new_play, active_new_internal = self.sub[i % len(self.sub)](play[i], sub_internals[i], util[i], grad[i], actual_time)
        new_play = play.copy()
        new_play[i] = active_new_play
        new_sub_internals = sub_internals.copy()
        new_sub_internals[i] = active_new_internal
        return new_play, ((i + 1) % n, new_sub_internals)


class MultiplicativeWeightUpdate(UpdateRule):
    InternalType = np.ndarray

    def __init__(self, lr: LearningRate, proj: Proj, optimism: float = 0):
        self.lr = lr
        self.proj = proj
        self.optimism = optimism

    def init_internal(self, n: int, m: int):
        return np.zeros(shape=(n, m))

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray, T: int) -> UpdateResult:
        lr = learning_rate(self.lr, T)
        new_internal = internal - grad
        new_internal -= np.max(new_internal, axis=1, keepdims=True)
        new_weights = np.exp(lr * (new_internal - self.optimism * grad))
        new_weights /= np.sum(new_weights, axis=1, keepdims=True)
        new_play = self.proj(new_weights)
        return new_play, new_internal


class GradientDescent(UpdateRule):
    def __init__(self, lr: LearningRate, proj: Proj):
        self.lr = lr
        self.proj = proj

    def init_internal(self, n, m):
        return None

    def __call__(self, play: np.ndarray, _, util: np.ndarray, grad: np.ndarray, T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        return self.proj(play - eta * grad), None


class OptimisticGradient(UpdateRule):
    def __init__(self, lr: LearningRate, proj: Proj):
        self.lr = lr
        self.proj = proj

    def init_internal(self, n, m) -> np.ndarray:
        return np.array([self.proj(np.ones(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        wholestep = internal
        # halfstep = play
        next_wholestep = self.proj(wholestep - eta * grad)
        next_halfstep = self.proj(next_wholestep - eta * grad)
        return next_halfstep, next_wholestep


class ExtraGradient(UpdateRule):
    def __init__(self, lr: LearningRate, proj: Proj):
        self.lr = lr
        self.proj = proj

    InternalType = Tuple[bool, np.ndarray]

    def init_internal(self, n, m) -> InternalType:
        return True, np.array([self.proj(np.ones(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: InternalType, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        if internal[0]:
            return self.proj(play - eta * grad), (not internal[0], play)
        else:
            return self.proj(internal[1] - eta * grad), (not internal[0], play)


class ODA_l2(UpdateRule):
    def __init__(self, lr: LearningRate, proj: Proj, optimism: float = 1):
        self.lr = lr
        self.proj = proj
        self.optimism = optimism

    def init_internal(self, n, m) -> np.ndarray:
        return np.array([(np.ones(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        total_util = internal + grad
        next_wholestep = self.proj(-eta * total_util)
        next_halfstep = self.proj(next_wholestep - self.optimism * eta * grad)
        return next_halfstep, total_util


class OFTRL_l2(UpdateRule):
    def __init__(self, lr: LearningRate, proj: Proj, optimism: float = 1):
        self.lr = lr
        self.proj = proj
        self.optimism = optimism

    def init_internal(self, n, m) -> np.ndarray:
        return np.array([(np.ones(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        total_util = internal + grad * (self.optimism + 1)
        return self.proj(-eta * total_util), internal + grad


class OFTRL(UpdateRule):
    """
    Ioannis Anagnostides, Gabriele Farina, Christian Kroer, Chung-Wei Lee, Haipeng Luo, and Tuomas Sandholm,
    “Uncoupled Learning Dynamics with $O(\log T)$ Swap Regret in Multiplayer Games.”
    arXiv, Apr. 24, 2022. doi: 10.48550/arXiv.2204.11417.
    """

    def __init__(self, lr: LearningRate, proj: Proj, barrier: Callable[[np.ndarray], float], optimism: float = 1):
        self.lr = lr
        self.proj = proj
        self.barrier = barrier
        self.optimism = optimism

    def init_internal(self, n, m) -> np.ndarray:
        return np.array([(np.ones(m)) for _ in range(n)])

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        eta = learning_rate(self.lr, T)
        total_util = internal + grad * (self.optimism + 1)
        new_play = np.zeros_like(play)
        for i in range(play.shape[0]):
            def f(x):
                return self.barrier(x) + eta * np.dot(total_util[i], x)

            new_play[i] = scipy.optimize.minimize(
                f, play[i], bounds=[(0, 1)] * play.shape[1],
                constraints=scipy.optimize.LinearConstraint(np.ones_like(play[i]), lb=1, ub=1)).x
        assert np.allclose(self.proj(new_play), new_play)
        return new_play, internal + grad


def log_barrier(x: np.ndarray) -> float:
    return -np.log(np.maximum(x, 1e-10)).sum()


def l2_regularizer(x: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(x, ord=2)) ** 2


class BlumMansour(UpdateRule):
    """
    Avrim Blum and Yishay Mansour, 
    “From External to Internal Regret,” 
    Journal of Machine Learning Research, vol. 8, no. 47, pp. 1307–1324, 2007.
    """

    def __init__(self, unit: UpdateRule):
        self.unit = unit

    def init_internal(self, n, m) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return [np.ones((n, m)) / m for _ in range(m)], [self.unit.init_internal(n, m) for _ in range(m)]

    def __call__(self, play: np.ndarray, internal: np.ndarray, util: np.ndarray, grad: np.ndarray,
                 T: int) -> UpdateResult:
        last_plays, unit_internals = internal
        new_plays = []
        new_unit_internals = []
        outputs = []
        for j in range(len(last_plays)):
            new_play, new_internal = self.unit(
                last_plays[j], unit_internals[j], util * play[:, j], np.einsum('ij,i->ij', grad, play[:, j]), T)
            new_plays.append(new_play)
            new_unit_internals.append(new_internal)
        for i in range(play.shape[0]):
            Q = np.array([new_plays[j][i]
                          for j in range(len(new_plays))])
            evals, evecs = np.linalg.eig(Q.T)
            evec1 = evecs[:, np.isclose(evals, 1)][:, 0]
            stationary = (evec1 / evec1.sum()).real
            outputs.append(stationary)
        return np.array(outputs), (new_plays, new_unit_internals)
