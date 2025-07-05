# FILE: src/lib/benchmarks/fn.py

"""
A Library of Benchmark Functions for Optimization Algorithms

This module provides a collection of classic optimization benchmark functions,
encapsulated as objects for easier use and metadata management. Each function
is implemented as a class inheriting from the `OptimizationFunction` base class.

The main export is `benchmark_library`, an instance of `BenchmarkLibrary`
that acts as a central registry for all defined functions.

Key Features:
- Object-oriented design: Each function is a self-contained object.
- Rich metadata: Each object stores its name, bounds, global minimum, and
  dimensionality constraints.
- Vectorized for performance: Functions use NumPy's array operations for speed.
- Central registry: Easily access or iterate over all available functions.

Example Usage:
--------------
from .fn import benchmark_library

# Get a specific function by name
sphere_func = benchmark_library.get("Sphere")

# Access its metadata
print(f"Bounds: {sphere_func.bounds}")
print(f"Global Minimum: {sphere_func.global_minimum_value}")

# Evaluate the function at a point
import numpy as np
point = np.array([1, 2, 3])
value = sphere_func(point)
print(f"Value at {point}: {value}")
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Iterator


class OptimizationFunction(ABC):
    """Abstract base class for an optimization benchmark function."""

    def __init__(
        self,
        name: str,
        bounds: Tuple[float, float],
        global_minimum_value: float,
        description: str,
        global_minimum_position: Optional[np.ndarray] = None,
        fixed_dims: Optional[int] = None,
    ):
        self.name = name
        self.bounds = bounds
        self.global_minimum_value = global_minimum_value
        self.description = description
        self.global_minimum_position = global_minimum_position
        self.fixed_dims = fixed_dims

    def __call__(self, x: np.ndarray) -> float:
        if self.fixed_dims is not None and len(x) != self.fixed_dims:
            raise ValueError(
                f"{self.name} function is only defined for {self.fixed_dims} dimensions, "
                f"but received {len(x)}."
            )
        return self._evaluate(x)

    @abstractmethod
    def _evaluate(self, x: np.ndarray) -> float:
        pass


# --- Concrete Function Implementations ---


class Sphere(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Sphere",
            bounds=(-5.12, 5.12),
            global_minimum_value=0.0,
            description="A simple, convex, and unimodal function.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        return np.sum(x**2)


class Rosenbrock(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Rosenbrock",
            bounds=(-5.0, 10.0),
            global_minimum_value=0.0,
            description="A challenging unimodal function with a narrow, parabolic valley.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Ackley(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Ackley",
            bounds=(-32.768, 32.768),
            global_minimum_value=0.0,
            description="Multi-modal with a nearly flat outer region and many local minima.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        n, a, b, c = len(x), 20, 0.2, 2 * np.pi
        return (
            -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
            - np.exp(np.sum(np.cos(c * x)) / n)
            + a
            + np.e
        )


class Griewank(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Griewank",
            bounds=(-600.0, 600.0),
            global_minimum_value=0.0,
            description="Multi-modal with many regularly distributed local minima.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        sum_term = np.sum(x**2) / 4000.0
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_term - prod_term


class Schwefel(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Schwefel",
            bounds=(-500.0, 500.0),
            global_minimum_value=0.0,
            description="Deceptive function where the global minimum is far from the local minima.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Rastrigin(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Rastrigin",
            bounds=(-5.12, 5.12),
            global_minimum_value=0.0,
            description="Highly multi-modal with a regular grid of local minima.",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        return 10 * len(x) + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Himmelblau(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Himmelblau",
            bounds=(-5.0, 5.0),
            global_minimum_value=0.0,
            fixed_dims=2,
            description="Multi-modal with four identical global minima (2D only).",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


class Easom(OptimizationFunction):
    def __init__(self):
        super().__init__(
            name="Easom",
            bounds=(-100.0, 100.0),
            global_minimum_value=-1.0,
            fixed_dims=2,
            description="Unimodal function with a very small area containing the global minimum (2D only).",
        )

    def _evaluate(self, x: np.ndarray) -> float:
        return (
            -np.cos(x[0])
            * np.cos(x[1])
            * np.exp(-((x[0] - np.pi) ** 2 + (x[1] - np.pi) ** 2))
        )


# --- Central Library/Registry ---


class BenchmarkLibrary:
    """A registry to hold and provide access to all benchmark functions."""

    def __init__(self, functions: List[OptimizationFunction]):
        self._functions: Dict[str, OptimizationFunction] = {
            f.name.lower(): f for f in functions
        }

    def get(self, name: str) -> OptimizationFunction:
        try:
            return self._functions[name.lower()]
        except KeyError:
            raise ValueError(
                f"Unknown function: '{name}'. Available: {self.list_functions()}"
            )

    def list_functions(self) -> List[str]:
        return [f.name for f in self._functions.values()]

    def __iter__(self) -> Iterator[OptimizationFunction]:
        return iter(self._functions.values())


benchmark_library = BenchmarkLibrary(
    [
        Sphere(),
        Rosenbrock(),
        Ackley(),
        Griewank(),
        Schwefel(),
        Rastrigin(),
        Himmelblau(),
        Easom(),
    ]
)
