# FILE: src/lib/model/ga.py (Refactored)

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Any
from lib.model.base import OptimizableAlgorithm


class GeneticAlgorithm(OptimizableAlgorithm):
    def __init__(
        self,
        pop_size: int,
        chromosome_length: int,
        fitness_function: Callable[[np.ndarray, Any], np.ndarray],
        problem_data: Any,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        maximize: bool = True,
        binary: bool = True,
    ):
        """
        Initializes a generic Genetic Algorithm for optimization problems.

        Args:
            pop_size (int): The size of the population.
            chromosome_length (int): The length of an individual's chromosome.
            fitness_function (Callable): Function to calculate fitness, accepting population and problem_data.
            problem_data (Any): Problem-specific data for the fitness function.
            mutation_rate (float): The probability of mutation for each gene.
            crossover_rate (float): The probability of crossover between two parents.
            elitism (int): Number of best individuals to carry over to the next generation.
            maximize (bool): True to maximize fitness, False to minimize.
            binary (bool): True for binary chromosomes (0,1), False for real-valued.
        """
        self.pop_size = pop_size
        self.chromosome_length = chromosome_length
        self.fitness_function = fitness_function
        self.problem_data = problem_data
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.maximize = maximize
        self.binary = binary

        if binary:
            self.population = np.random.randint(0, 2, (pop_size, chromosome_length))
        else:
            self.population = np.random.rand(pop_size, chromosome_length)

        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome = None
        self.best_fitness = -np.inf if maximize else np.inf

    def _calculate_fitness(self) -> np.ndarray:
        return self.fitness_function(self.population, self.problem_data)

    def _select_parents(self, fitness: np.ndarray) -> np.ndarray:
        """Selects parents using tournament selection."""
        tournament_size = 3
        selected_indices = []
        for _ in range(self.pop_size):
            tournament_indices = np.random.randint(0, self.pop_size, tournament_size)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[
                np.argmax(tournament_fitness)
                if self.maximize
                else np.argmin(tournament_fitness)
            ]
            selected_indices.append(winner_idx)
        return np.array(selected_indices)

    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Performs crossover between pairs of parents."""
        offspring = np.zeros_like(self.population)

        # Elitism
        if self.elitism > 0:
            fitness = self._calculate_fitness()
            elite_indices = np.argsort(fitness)
            if self.maximize:
                elite_indices = elite_indices[::-1]
            offspring[: self.elitism] = self.population[elite_indices[: self.elitism]]

        # Crossover for the rest
        for i in range(self.elitism, self.pop_size, 2):
            parent1, parent2 = (
                self.population[parents[i]],
                self.population[parents[(i + 1) % len(parents)]],
            )
            if np.random.rand() < self.crossover_rate and i + 1 < self.pop_size:
                point = np.random.randint(1, self.chromosome_length)
                offspring[i, :point], offspring[i, point:] = (
                    parent1[:point],
                    parent2[point:],
                )
                offspring[i + 1, :point], offspring[i + 1, point:] = (
                    parent2[:point],
                    parent1[point:],
                )
            else:
                offspring[i], offspring[i + 1] = parent1.copy(), parent2.copy()
        return offspring

    def _mutate(self, population: np.ndarray) -> np.ndarray:
        """Applies mutation to the population."""
        mutation_mask = np.random.rand(*population.shape) < self.mutation_rate
        if self.elitism > 0:
            mutation_mask[: self.elitism] = False

        mutated_population = population.copy()
        if self.binary:
            mutated_population[mutation_mask] = 1 - mutated_population[mutation_mask]
        else:
            noise = np.random.normal(0, 0.1, population.shape)
            mutated_population[mutation_mask] += noise[mutation_mask]
            np.clip(mutated_population, 0, 1, out=mutated_population)
        return mutated_population

    def evolve(self):
        """Executes one generation of the genetic algorithm."""
        fitness = self._calculate_fitness()
        self.avg_fitness_history.append(np.mean(fitness))
        best_idx_current = np.argmax(fitness) if self.maximize else np.argmin(fitness)
        current_best_fitness = fitness[best_idx_current]
        self.best_fitness_history.append(current_best_fitness)

        if (self.maximize and current_best_fitness > self.best_fitness) or (
            not self.maximize and current_best_fitness < self.best_fitness
        ):
            self.best_fitness = current_best_fitness
            self.best_chromosome = self.population[best_idx_current].copy()

        parents_indices = self._select_parents(fitness)
        offspring = self._crossover(parents_indices)
        self.population = self._mutate(offspring)

    def run(self, generations: int, verbose: bool = True) -> Tuple[np.ndarray, float]:
        """Runs the GA for a number of generations."""
        for gen in range(generations):
            self.evolve()
            if verbose and ((gen + 1) % 10 == 0 or gen == 0 or gen == generations - 1):
                print(
                    f"Generation {gen + 1}: Best Fitness = {self.best_fitness_history[-1]:.2f}, "
                    f"Avg Fitness = {self.avg_fitness_history[-1]:.2f}"
                )
        if verbose:
            print(f"\nBest solution found:\n  Fitness: {self.best_fitness:.2f}")
            print(f"  Chromosome: {self.best_chromosome.astype(int)}")
        return self.best_chromosome, self.best_fitness

    def plot_history(self, title: str = "Fitness Evolution"):
        plt.figure(figsize=(10, 5))
        plt.plot(self.best_fitness_history, "b-", label="Best Fitness")
        plt.plot(self.avg_fitness_history, "r-", label="Average Fitness")
        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()
