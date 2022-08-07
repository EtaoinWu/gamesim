from typing import Sequence, Literal
from base import *
from game import *
from rules import *
import itertools

class GameSim:
    def __init__(self, game : GameBase, rule : UpdateRule):
        self.game = game
        self.rule = rule
        self.n = game.n
        self.m = game.m
        self.state = np.zeros((self.n, self.m))
        self.internal = rule.init_internal(self.n, self.m)
        self.trajectory : List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    
    def play(self, steps : int) -> None:
        for _ in range(steps):
            util, ngrad = self.game(*self.state)
            grad = -ngrad
            self.trajectory.append((self.state.copy(), util, grad))
            self.state, self.internal = self.rule(self.state, self.internal, util, grad)

    RewriteFunc = List[int]
    RewriteType = Union[Sequence[RewriteFunc], None, Literal["swap"], Literal["internal"]]
    # RewriteType = Union[Sequence[RewriteFunc], None, str] # for colab 3.77

    def rewrites(self, f : RewriteType) -> Sequence[RewriteFunc]:
        if f is None:
            return [[i] * self.m for i in range(self.m)]
        elif f == "swap":
            return [[t[i] for i in range(self.m)] for t in itertools.product(list(range(self.m)), repeat=self.m)]
        elif f == "internal":
            return [[j if x == i else x for x in range(self.m)] for i in range(self.m) for j in range(self.m)]
        else:
            return f

    @staticmethod
    def rewrite_single_state(state : np.ndarray, rewrite_function : RewriteFunc) -> np.ndarray:
        new_state = np.zeros_like(state)
        for i in range(state.shape[0]):
            new_state[rewrite_function[i]] += state[i]
        return new_state

    @staticmethod
    def rewrite_state(state : np.ndarray, i : int, rewrite_function : RewriteFunc) -> np.ndarray:
        new_state = state.copy()
        new_state[i] = GameSim.rewrite_single_state(state[i], rewrite_function)
        return new_state

    def individual_regret(self, i : int, rewrite_functions : RewriteType = None) -> float:
        return max([
            sum([
                self.game.value(*GameSim.rewrite_state(state, i, rewrite))[i] - util[i]
                for state, util, _ in self.trajectory
            ])
            for rewrite in self.rewrites(rewrite_functions)
        ])

    def regret(self, rewrite_functions : RewriteType = None) -> np.ndarray:
        return np.array([self.individual_regret(i, rewrite_functions) for i in range(self.n)])
    
    def social_regret(self, rewrite_functions : RewriteType = None) -> np.number:
        return self.regret(rewrite_functions).sum()
    
    def max_regret(self, rewrite_functions : RewriteType = None) -> np.number:
        return self.regret(rewrite_functions).max()