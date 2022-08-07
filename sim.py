from base import *
from game import *
from rules import *

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
            util, grad = self.game(*self.state)
            self.trajectory.append((self.state.copy(), util, grad))
            self.state, self.internal = self.rule(self.state, self.internal, util, grad)
