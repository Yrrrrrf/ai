# FILE: src/lib/benchmark/configs.py

"""
Central Configuration for Algorithm Benchmarking

This module contains predefined configurations for various algorithms,
including hyperparameter sets for comparison tests. This centralized approach
makes it easy to manage and extend algorithm settings without modifying the
benchmark logic itself.
"""

# Import the algorithm classes to link them in the config
from lib.model.ga import GeneticAlgorithm
from ..model.pso import PSO


DEFAULT_ALGO_PARAMS = {
    "PSO": {
        "num_particles": 30,
        "w": 0.5,
        "c1": 1.5,
        "c2": 1.5,
        "max_iterations": 100,
    },
    "GeneticAlgorithm": {
        "pop_size": 100,
        "mutation_rate": 0.01,
        "crossover_rate": 0.7,
        "elitism": 2,
        "generations": 100,
    },
}

# This dictionary is the single source of truth for hyperparameter sets.
ALGORITHM_CONFIGS = {
    "PSO": {
        "class": PSO,
        "parameter_sets": [
            {"w": 0.5, "c1": 1.5, "c2": 1.5, "label": "Default"},
            {"w": 0.7, "c1": 1.0, "c2": 2.0, "label": "Social Dominant"},
            {"w": 0.7, "c1": 2.0, "c2": 1.0, "label": "Cognitive Dominant"},
            {"w": 0.9, "c1": 1.5, "c2": 1.5, "label": "High Inertia (Exploration)"},
            {"w": 0.3, "c1": 1.5, "c2": 1.5, "label": "Low Inertia (Exploitation)"},
        ],
    },
    "Genetic Algorithm": {
        "class": GeneticAlgorithm,
        # Parameters for GA are different, but the structure is the same.
        "parameter_sets": [
            {"mutation_rate": 0.01, "crossover_rate": 0.7, "label": "Standard"},
            {"mutation_rate": 0.05, "crossover_rate": 0.8, "label": "High Mutation/Crossover"},
            {"mutation_rate": 0.005, "crossover_rate": 0.5, "label": "Low Mutation/Crossover"},
            {"mutation_rate": 0.02, "crossover_rate": 0.9, "label": "High Exploration"},
        ]
    }
}
