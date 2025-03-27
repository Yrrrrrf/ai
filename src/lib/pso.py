"""
# Particle Swarm Optimization Implementation

This notebook demonstrates how to implement and apply the Particle Swarm Optimization (PSO) algorithm
to solve optimization problems.
"""

# Required imports
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Callable
from matplotlib.animation import FuncAnimation


class PSO:
    def __init__(
        self,
        num_particles: int,
        dimensions: int,
        fitness_function: Callable[[np.ndarray], float],
        bounds: Tuple[float, float],
        w: float = 0.5,  # Inertia weight
        c1: float = 1.5,  # Cognitive weight
        c2: float = 1.5,  # Social weight
        max_iterations: int = 100,
        minimize: bool = True,
    ):
        """
        Initialize the PSO algorithm.

        Args:
            num_particles: Number of particles in the swarm
            dimensions: Dimensionality of the search space
            fitness_function: Function to optimize
            bounds: Tuple of (min, max) defining the search space
            w: Inertia weight
            c1: Cognitive weight (personal best influence)
            c2: Social weight (global best influence)
            max_iterations: Maximum number of iterations
            minimize: If True, minimize the function; otherwise, maximize
        """
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.minimize = minimize

        # Initialize history tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.positions_history = []

        # Initialize particles
        self._initialize_swarm()

    def _initialize_swarm(self):
        """Initialize the particle positions and velocities."""
        # Initialize positions randomly within bounds
        low, high = self.bounds
        self.positions = np.random.uniform(
            low, high, (self.num_particles, self.dimensions)
        )

        # Initialize velocities randomly within a fraction of the bounds range
        velocity_range = (high - low) * 0.1
        self.velocities = np.random.uniform(
            -velocity_range, velocity_range, (self.num_particles, self.dimensions)
        )

        # Evaluate initial positions
        self.fitness_values = np.array(
            [self.fitness_function(pos) for pos in self.positions]
        )

        # Initialize personal best positions and fitness values
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = self.fitness_values.copy()

        # Initialize global best
        if self.minimize:
            self.global_best_idx = np.argmin(self.fitness_values)
        else:
            self.global_best_idx = np.argmax(self.fitness_values)

        self.global_best_position = self.positions[self.global_best_idx].copy()
        self.global_best_fitness = self.fitness_values[self.global_best_idx]

        # Save initial state to history
        self.positions_history.append(self.positions.copy())
        self.best_fitness_history.append(self.global_best_fitness)
        self.avg_fitness_history.append(np.mean(self.fitness_values))

    def update_velocity(self):
        """Update velocities of all particles."""
        # Generate random values for stochastic components
        r1 = np.random.random((self.num_particles, self.dimensions))
        r2 = np.random.random((self.num_particles, self.dimensions))

        # Calculate cognitive and social components
        cognitive_component = (
            self.c1 * r1 * (self.personal_best_positions - self.positions)
        )
        social_component = self.c2 * r2 * (self.global_best_position - self.positions)

        # Update velocities using PSO formula
        self.velocities = (
            self.w * self.velocities + cognitive_component + social_component
        )

    def update_position(self):
        """Update positions of all particles based on velocities."""
        self.positions += self.velocities  # Update positions
        self.positions = np.clip(
            self.positions, self.bounds[0], self.bounds[1]
        )  # Enforce bounds

    def evaluate_fitness(self):
        """Evaluate fitness of all particles at their current positions."""
        self.fitness_values = np.array(
            [self.fitness_function(pos) for pos in self.positions]
        )

        # Update personal bests
        if self.minimize:
            improved = self.fitness_values < self.personal_best_fitness
        else:
            improved = self.fitness_values > self.personal_best_fitness

        self.personal_best_positions[improved] = self.positions[improved]
        self.personal_best_fitness[improved] = self.fitness_values[improved]

        # Update global best
        if self.minimize:
            current_best_idx = np.argmin(self.fitness_values)
            if self.fitness_values[current_best_idx] < self.global_best_fitness:
                self.global_best_idx = current_best_idx
                self.global_best_position = self.positions[current_best_idx].copy()
                self.global_best_fitness = self.fitness_values[current_best_idx]
        else:
            current_best_idx = np.argmax(self.fitness_values)
            if self.fitness_values[current_best_idx] > self.global_best_fitness:
                self.global_best_idx = current_best_idx
                self.global_best_position = self.positions[current_best_idx].copy()
                self.global_best_fitness = self.fitness_values[current_best_idx]

    def optimize(self, verbose=True):
        """Run the PSO optimization process."""
        for iteration in range(self.max_iterations):
            self.update_velocity()  # Update velocities
            self.update_position()  # Update positions
            self.evaluate_fitness()  # Evaluate fitness

            # Save current state to history
            self.positions_history.append(self.positions.copy())
            self.best_fitness_history.append(self.global_best_fitness)
            self.avg_fitness_history.append(np.mean(self.fitness_values))

            # Print progress if verbose
            if verbose and (
                iteration % 10 == 0 or iteration == self.max_iterations - 1
            ):
                print(
                    f"Iteration {iteration + 1}: Best fitness = {self.global_best_fitness:.4f}, "
                    f"Avg fitness = {self.avg_fitness_history[-1]:.4f}"
                )

        return self.global_best_position, self.global_best_fitness

    def plot_history(self, title="PSO Optimization"):
        """Plot the history of fitness values."""
        plt.figure(figsize=(10, 5))
        iterations = range(1, len(self.best_fitness_history) + 1)

        plt.plot(iterations, self.best_fitness_history, "b-", label="Best fitness")
        plt.plot(iterations, self.avg_fitness_history, "r-", label="Average fitness")

        plt.title(title)
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ^ THIS FN ONLY WORKS WHEN CALLED FROM A SCRIPT!
    # ^ THIS FN WOULDN'T WORK IF CALLED FROM A JUPYTER NOTEBOOK!
    # todo: add save_path parameter, and save the plot to that path
    # todo: Improve the fn to make it work in both cases
    def visualize_particles_2d(self, interval=20, save_path=None):
        """
        Create an animation of particles moving in 2D space.
        Only works for 2D optimization problems.
        """
        if self.dimensions != 2:
            print("Visualization only available for 2D problems.")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a contour plot of the fitness function
        x = np.linspace(self.bounds[0], self.bounds[1], 100)
        y = np.linspace(self.bounds[0], self.bounds[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = self.fitness_function(np.array([X[i, j], Y[i, j]]))

        contour = ax.contourf(X, Y, Z, 50, cmap="viridis", alpha=0.6)
        plt.colorbar(contour, ax=ax)

        # Plot the global optimum
        ax.scatter(
            self.global_best_position[0],
            self.global_best_position[1],
            c="red",
            marker="*",
            s=200,
            label="Global Best",
        )

        # Initialize particle scatter plot
        particles_scatter = ax.scatter([], [], c="white", edgecolors="black", s=50)

        # Set plot boundaries and labels
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[0], self.bounds[1])
        ax.set_title("Particle Swarm Optimization")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()

        # Animation update function
        def update(frame):
            particles_scatter.set_offsets(self.positions_history[frame])
            ax.set_title(f"Particle Swarm Optimization - Iteration {frame}")
            return (particles_scatter,)

        # Create animation
        animation = FuncAnimation(
            fig,
            update,
            frames=len(self.positions_history),
            interval=interval,
            blit=True,
        )

        # Save animation if path is provided
        if save_path:
            animation.save(save_path, writer="pillow")

        plt.tight_layout()
        plt.show()

        return animation
