# FILE: src/lib/benchmark/problems.py
# (This file REPLACES fn.py)

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple

# =================================================================== #
# 1. THE ABSTRACT PROBLEM DEFINITION (The core of the new system)
# =================================================================== #


class BenchmarkProblem(ABC):
    """Abstract base class for a benchmark problem."""

    def __init__(self, name: str, minimize: bool = True, tags: list[str] = None):
        self.name = name
        self.minimize = minimize
        self.tags = tags if tags else []

    @abstractmethod
    def get_config_for_algorithm(self, algo_name: str) -> Dict[str, Any]:
        """Returns the specific configuration an algorithm needs to solve this problem."""
        pass

    @abstractmethod
    def fitness_function(self, solution: np.ndarray, problem_data: Any = None) -> float:
        """The fitness function. The 'problem_data' is for complex cases like Knapsack."""
        pass


# =================================================================== #
# 2. THE LOGIC FROM YOUR fn.py (Now as plain, reusable functions)
# =================================================================== #


def sphere_func(x: np.ndarray) -> float:
    return np.sum(x**2)


def rosenbrock_func(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def ackley_func(x: np.ndarray) -> float:
    n, a, b, c = len(x), 20, 0.2, 2 * np.pi
    return (
        -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(c * x)) / n)
        + a
        + np.e
    )


def griewank_func(x: np.ndarray) -> float:
    sum_term = np.sum(x**2) / 4000.0
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term


def schwefel_func(x: np.ndarray) -> float:
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin_func(x: np.ndarray) -> float:
    return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


def himmelblau_func(x: np.ndarray) -> float:
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def easom_func(x: np.ndarray) -> float:
    return (
        -np.cos(x[0])
        * np.cos(x[1])
        * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
    )


# =================================================================== #
# 3. CONCRETE PROBLEM IMPLEMENTATIONS (The wrappers)
# =================================================================== #


class ContinuousProblem(BenchmarkProblem):
    """A generic wrapper for any continuous optimization function."""

    def __init__(
        self,
        name: str,
        func: callable,
        bounds: Tuple[float, float],
        dimensions: int,
        tags: list[str] = None,
    ):
        super().__init__(name=name, minimize=True, tags=(tags or []) + ["continuous"])
        self._func = func
        self.bounds = bounds
        self.dimensions = dimensions

    def get_config_for_algorithm(self, algo_name: str) -> Dict[str, Any]:
        # Most swarm/evolutionary algorithms for continuous problems need these same parameters.
        return {
            "fitness_function": self.fitness_function,
            "dimensions": self.dimensions,
            "bounds": self.bounds,
            "minimize": self.minimize,
        }

    def fitness_function(self, solution: np.ndarray, problem_data: Any = None) -> float:
        return self._func(solution)


class KnapsackProblem(BenchmarkProblem):
    """The Knapsack problem, a combinatorial optimization example."""

    def __init__(self):
        super().__init__(
            name="Knapsack", minimize=False, tags=["combinatorial", "binary"]
        )
        self.problem_data = {
            "items": [(10, 60), (20, 100), (30, 120), (15, 70), (25, 110)],
            "max_capacity": 150,
            "item_count": 5,
        }

    def get_config_for_algorithm(self, algo_name: str) -> Dict[str, Any]:
        if algo_name == "GeneticAlgorithm":
            return {
                "fitness_function": self.fitness_function,
                "chromosome_length": self.problem_data["item_count"],
                "problem_data": self.problem_data,
                "binary": True,
                "maximize": not self.minimize,
            }
        # A standard PSO can't solve this, so we raise an error.
        raise NotImplementedError(
            f"The Knapsack problem is not configured for the '{algo_name}' algorithm."
        )

    def fitness_function(self, population: np.ndarray, problem_data: Any) -> np.ndarray:
        weights = np.array([item[0] for item in problem_data["items"]])
        values = np.array([item[1] for item in problem_data["items"]])

        # This function is called from GA, which evaluates the whole population at once
        if population.ndim == 1:  # If called with a single solution
            population = population.reshape(1, -1)

        total_values = np.sum(population * values, axis=1)
        total_weights = np.sum(population * weights, axis=1)
        total_values[total_weights > problem_data["max_capacity"]] = 0
        return total_values


# =================================================================== #
# 4. THE CENTRAL REGISTRY (Now a simple list of problem instances)
# =================================================================== #

benchmark_problem_library = [
    # We instantiate a generic ContinuousProblem for each function.
    # The 'fixed_dims' logic is now handled by explicitly setting dimensions.
    ContinuousProblem(
        name="Sphere (2D)",
        func=sphere_func,
        bounds=(-5.12, 5.12),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Rosenbrock (2D)",
        func=rosenbrock_func,
        bounds=(-5.0, 10.0),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Ackley (2D)",
        func=ackley_func,
        bounds=(-32.768, 32.768),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Griewank (2D)",
        func=griewank_func,
        bounds=(-600.0, 600.0),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Schwefel (2D)",
        func=schwefel_func,
        bounds=(-500.0, 500.0),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Rastrigin (2D)",
        func=rastrigin_func,
        bounds=(-5.12, 5.12),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Himmelblau (2D)",
        func=himmelblau_func,
        bounds=(-5.0, 5.0),
        dimensions=2,
        tags=["2d"],
    ),
    ContinuousProblem(
        name="Easom (2D)",
        func=easom_func,
        bounds=(-100.0, 100.0),
        dimensions=2,
        tags=["2d"],
    ),
    # We can also add higher-dimensional problems easily
    ContinuousProblem(
        name="Sphere (10D)", func=sphere_func, bounds=(-5.12, 5.12), dimensions=10
    ),
    # And our combinatorial problem lives happily in the same list!
    KnapsackProblem(),
]
