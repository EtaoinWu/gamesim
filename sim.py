from typing import Sequence, Literal
from base import *
from game import *
from rules import *
import itertools


RewriteFunc = List[int]
RewriteType = Union[Sequence[RewriteFunc],
                    None, Literal["swap"], Literal["internal"]]


class GameSimBase(ABC):
    @abstractmethod
    def __init__(self, game: GameBase, rule: UpdateRule, regret_types: List[RewriteType] = []):
        self.game = game
        self.rule = rule
        self.n = game.n
        self.m = game.m


class GameSim(GameSimBase):
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

    class RegretRecorder:
        def __init__(self, game_sim: GameSimBase, rewrite_functions: Sequence[RewriteFunc]):
            self.game_sim = game_sim
            self.rewrite_functions = rewrite_functions
            # self.regret_trajectory[player][step][rewrite_func] = float
            self.regret_trajectory: List[List[np.ndarray]] = [
                [] for _ in range(game_sim.n)]

        def play(self, state: np.ndarray, util: np.ndarray):
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

        def social(self) -> np.ndarray:
            return np.array(self().sum(axis=0))

        def max(self) -> np.ndarray:
            return np.array(self().max(axis=0))

    def __init__(self, game: GameBase, rule: UpdateRule, regret_types=[]):
        self.game = game
        self.rule = rule
        self.n = game.n
        self.m = game.m
        self.state = np.zeros((self.n, self.m))
        self.internal = rule.init_internal(self.n, self.m)
        self.trajectory: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.regret_recorders = {rewrite_type: self.RegretRecorder(self, self.rewrites(rewrite_type)) for rewrite_type
                                 in regret_types}

    def play(self, steps: int) -> None:
        for _ in range(steps):
            util, ngrad = self.game(*self.state)
            grad = -ngrad
            self.trajectory.append((self.state.copy(), util, grad))
            for _, regret_recorder in self.regret_recorders.items():
                regret_recorder.play(self.state, util)
            self.state, self.internal = self.rule(
                self.state, self.internal, util, grad)

    def rewrites(self, f: RewriteType) -> Sequence[RewriteFunc]:
        if f is None:
            return [[i] * self.m for i in range(self.m)]
        elif f == "swap":
            return [[t[i] for i in range(self.m)] for t in itertools.product(list(range(self.m)), repeat=self.m)]
        elif f == "internal":
            return [[j if x == i else x for x in range(self.m)] for i in range(self.m) for j in range(self.m)]
        else:
            return f

    def individual_regret(self, i: int, rewrite_functions: RewriteType = None) -> float:
        return max([
            sum([
                self.game.value(
                    *GameSim.rewrite_state(state, i, rewrite))[i] - util[i]
                for state, util, _ in self.trajectory
            ])
            for rewrite in self.rewrites(rewrite_functions)
        ])

    def regret(self, rewrite_functions: RewriteType = None) -> np.ndarray:
        return np.array([self.individual_regret(i, rewrite_functions) for i in range(self.n)])

    def social_regret(self, rewrite_functions: RewriteType = None) -> np.number:
        return self.regret(rewrite_functions).sum()

    def max_regret(self, rewrite_functions: RewriteType = None) -> np.number:
        return self.regret(rewrite_functions).max()
