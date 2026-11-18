from __future__ import annotations

class EarlyStopping:
    def __init__(self, patience: int = 8, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float):
        if self.best is None:
            self.best = value
            return True

        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False
