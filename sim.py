from typing import Sequence, Literal, NamedTuple
from base import *
from game import *
from rules import *
import itertools

RewriteFunc = List[int]
RewriteType = Union[Sequence[RewriteFunc],
                    None, Literal["swap"], Literal["internal"]]


class GameSimBase(ABC):
    @abstractmethod
    def __init__(self, game: GameBase, rule: UpdateRule, proj: Proj, regret_types: List[RewriteType] = []):
        self.game = game
        self.rule = rule
        self.proj = proj
        self.n = game.n
        self.m = game.m


class GameSim(GameSimBase):
    class Trajectory(NamedTuple):
        state: np.ndarray
        util: np.ndarray
        grad: np.ndarray
        internal: Any

    @staticmethod
    def rewrite_single_state(state: np.ndarray, rewrite_function: RewriteFunc) -> np.ndarray:
        new_state = np.zeros_like(state)
        for i in range(state.shape[0]):
            new_state[rewrite_function[i]] += state[i]
        return new_state

    @staticmethod
    def rewrite_state(state: np.ndarray, i: int, rewrite_function: RewriteFunc) -> np.ndarray:
        new_state = state.copy()
        new_state[i] = GameSim.rewrite_single_state(state[i], rewrite_function)
        return new_state

    class RegretRecorderBase(ABC):
        @abstractmethod
        def play(self, state: np.ndarray, util: np.ndarray, grad: np.ndarray):
            pass

        @abstractmethod
        def __call__(self) -> np.ndarray:
            pass

        @abstractmethod
        def x_star(self) -> np.ndarray:
            pass

        def social(self) -> np.ndarray:
            return np.array(self().sum(axis=0))

        def max(self) -> np.ndarray:
            return np.array(self().max(axis=0))

    class RegretRecorder(RegretRecorderBase):
        def __init__(self, game_sim: GameSimBase, rewrite_functions: Sequence[RewriteFunc], _: Proj):
            self.game_sim = game_sim
            self.rewrite_functions = rewrite_functions
            # self.regret_trajectory[player][step][rewrite_func] = float
            self.regret_trajectory: List[List[np.ndarray]] = [
                [] for _ in range(game_sim.n)]

        def play(self, state: np.ndarray, util: np.ndarray, _: np.ndarray):
            for player in range(self.game_sim.n):
                l = []
                for rewrite in self.rewrite_functions:
                    l.append(
                        self.game_sim.game.value(
                            *GameSim.rewrite_state(state, player, rewrite))[player]
                        - util[player]
                    )
                self.regret_trajectory[player].append(np.array(l))

        def __call__(self) -> np.ndarray:
            return np.array(np.array(self.regret_trajectory).cumsum(axis=1).max(axis=2))

        def x_star(self) -> np.ndarray:
            temp = np.array(self.regret_trajectory).cumsum(axis=1)
            return proj_sim((temp == temp.max(axis=-1, keepdims=True)).astype(float))

    class NormedRegretRecorder(RegretRecorderBase):
        def __init__(self, game_sim: GameSimBase, _: Sequence[RewriteFunc], proj: Proj):
            self.game_sim = game_sim
            self.proj = proj
            # self.grad_trajectory[step][player] = grad
            self.grad_trajectory: List[np.ndarray] = []
            # self.util_trajectory[step][player] = float
            self.util_trajectory: List[np.ndarray] = []

        def play(self, _1: np.ndarray, util: np.ndarray, grad: np.ndarray):
            self.grad_trajectory.append(grad)
            self.util_trajectory.append(util)

        def __call__(self) -> np.ndarray:
            temp = - np.swapaxes(
                np.array(self.grad_trajectory), 0, 1).cumsum(axis=1)
            best_pure = self.proj.best_pure(temp)
            return np.einsum('ijk,ijk->ij', temp, best_pure) - np.swapaxes(np.array(self.util_trajectory), 0, 1).cumsum(axis=1)

        def regret_part1(self) -> np.ndarray:
            temp = - np.swapaxes(
                np.array(self.grad_trajectory), 0, 1).cumsum(axis=1)
            best_pure = self.proj.best_pure(temp)
            return np.einsum('ijk,ijk->ij', temp, best_pure)

        def x_star(self) -> np.ndarray:
            temp = - np.swapaxes(
                np.array(self.grad_trajectory), 0, 1).cumsum(axis=1)
            return self.proj.best_pure(temp)

    regret_recorders: dict[Sequence[RewriteFunc] | str | None, RegretRecorderBase]

    def __init__(self, game: GameBase, rule: UpdateRule, proj: Proj, regret_types=[]):
        self.game = game
        self.rule = rule
        self.proj = proj
        self.n = game.n
        self.m = game.m
        self.state = np.ones((self.n, self.m)) / self.m
        self.internal = rule.init_internal(self.n, self.m)
        self.trajectory: List[GameSim.Trajectory] = []
        self.regret_types = regret_types
        self.normed_regret_recorder = GameSim.NormedRegretRecorder(
            self, self.regret_types, self.proj
        )
        self.regret_recorders = {rewrite_type: GameSim.RegretRecorder(self, self.rewrites(rewrite_type), proj) for rewrite_type
                                 in regret_types}
        self.regret_recorders['normed'] = self.normed_regret_recorder
        self.regret_types = self.regret_recorders.keys()

    def play(self, steps: int) -> None:
        for _ in range(steps):
            util, ngrad = self.game(*self.state)
            grad = -ngrad
            self.trajectory.append(GameSim.Trajectory(
                self.state.copy(), util, grad, self.internal))
            for _, regret_recorder in self.regret_recorders.items():
                regret_recorder.play(self.state, util, grad)
            self.state, self.internal = self.rule(
                self.state, self.internal, util, grad, len(self.trajectory))

    def rewrites(self, f: RewriteType) -> Sequence[RewriteFunc]:
        if f is None:
            return [[i] * self.m for i in range(self.m)]
        elif f == "swap":
            return [[t[i] for i in range(self.m)] for t in itertools.product(list(range(self.m)), repeat=self.m)]
        elif f == "internal":
            return [[j if x == i else x for x in range(self.m)] for i in range(self.m) for j in range(self.m)]
        else:
            return f

    # def individual_regret(self, i: int, rewrite_functions: RewriteType = None) -> float:
    #     return max([
    #         sum([
    #             self.game.value(
    #                 *GameSim.rewrite_state(state, i, rewrite))[i] - util[i]
    #             for state, util, *_ in self.trajectory
    #         ])
    #         for rewrite in self.rewrites(rewrite_functions)
    #     ])
    #
    # def regret(self, rewrite_functions: RewriteType = None) -> np.ndarray:
    #     return np.array([self.individual_regret(i, rewrite_functions) for i in range(self.n)])
    #
    # def social_regret(self, rewrite_functions: RewriteType = None) -> np.number:
    #     return self.regret(rewrite_functions).sum()
    #
    # def max_regret(self, rewrite_functions: RewriteType = None) -> np.number:
    #     return self.regret(rewrite_functions).max()

    def path_length(self, order: Union[int, float, None] = None, power: float = 1):
        return np.cumsum(
            np.linalg.norm(
                np.diff(np.array([a for a, *_ in self.trajectory]), axis=0),
                axis=-1,
                ord=order) ** power, axis=0)

    def grad_path_length(self, order: Union[int, float, None] = None, power: float = 1):
        return np.cumsum(
            np.linalg.norm(
                np.diff(np.array([a.grad for a in self.trajectory]), axis=0),
                axis=-1,
                ord=order) ** power, axis=0)
