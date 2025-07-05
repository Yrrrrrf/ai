# FILE: src/lib/model/base.py
from abc import ABC, abstractmethod

class OptimizableAlgorithm(ABC):
    best_fitness_history: list

    @abstractmethod
    def run(self, verbose: bool = False):
        """Runs the optimization process."""
        pass