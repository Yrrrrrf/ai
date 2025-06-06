import torch
import matplotlib.pyplot as plt
from typing import Tuple, Callable, Any


class GeneticAlgorithm:
    def __init__(
        self,
        pop_size: int,
        chromosome_length: int,
        fitness_function: Callable[[torch.Tensor, Any], torch.Tensor],
        problem_data: Any,
        mutation_rate: float = 0.01,
        crossover_rate: float = 0.7,
        elitism: int = 2,
        maximize: bool = True,
        binary: bool = True,
    ):
        """
        Inicializa el algoritmo genético genérico para problemas de optimización

        Args:
            pop_size: Tamaño de la población
            chromosome_length: Longitud del cromosoma
            fitness_function: Función que calcula el fitness (debe aceptar población y problem_data)
            problem_data: Datos específicos del problema para la función de fitness
            mutation_rate: Tasa de mutación
            crossover_rate: Probabilidad de cruce
            elitism: Número de mejores individuos que pasan directamente a la siguiente generación
            maximize: True si se busca maximizar el fitness, False si se busca minimizar
            binary: True si los cromosomas son binarios (0,1), False si son reales
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

        # Inicializar población aleatoria
        if binary:
            self.population = torch.randint(
                0, 2, (pop_size, chromosome_length), dtype=torch.float32
            )
        else:
            self.population = torch.rand(
                (pop_size, chromosome_length), dtype=torch.float32
            )

        # Historiales para seguimiento
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_chromosome = None
        self.best_fitness = float("-inf") if maximize else float("inf")

    def calculate_fitness(self, population: torch.Tensor) -> torch.Tensor:
        """
        Calcula el valor de fitness para cada individuo de la población
        usando la función de fitness proporcionada
        """
        return self.fitness_function(population, self.problem_data)

    def select_parents(self, fitness: torch.Tensor) -> torch.Tensor:
        """
        Selecciona padres para reproducción usando selección por torneo

        Args:
            fitness: Tensor con los valores de fitness de cada individuo

        Returns:
            Índices de los padres seleccionados
        """
        tournament_size = 3
        selected_indices = []

        for _ in range(self.pop_size):
            # Seleccionar aleatoriamente individuos para el torneo
            tournament_indices = torch.randint(0, self.pop_size, (tournament_size,))
            tournament_fitness = fitness[tournament_indices]

            # Elegir al mejor o peor del torneo según si maximizamos o minimizamos
            if self.maximize:
                winner_idx = tournament_indices[torch.argmax(tournament_fitness)]
            else:
                winner_idx = tournament_indices[torch.argmin(tournament_fitness)]

            selected_indices.append(winner_idx)

        return torch.tensor(selected_indices)

    def crossover(self, parents: torch.Tensor) -> torch.Tensor:
        """
        Realiza el cruce entre pares de padres para crear nueva descendencia
        """
        offspring = torch.zeros_like(self.population)

        # Preservar los mejores individuos (elitismo)
        if self.elitism > 0:
            fitness = self.calculate_fitness(self.population)
            if self.maximize:
                elite_indices = torch.argsort(fitness, descending=True)[: self.elitism]
            else:
                elite_indices = torch.argsort(fitness, descending=False)[: self.elitism]

            offspring[: self.elitism] = self.population[elite_indices]

        # Cruce para el resto de la población
        for i in range(self.elitism, self.pop_size, 2):
            # Seleccionar dos padres
            parent1_idx = parents[i % len(parents)]
            parent2_idx = parents[(i + 1) % len(parents)]

            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]

            # Aplicar cruce con probabilidad crossover_rate
            if torch.rand(1).item() < self.crossover_rate:
                # Punto de cruce aleatorio
                crossover_point = torch.randint(1, self.chromosome_length, (1,)).item()

                # Crear descendencia
                if i < self.pop_size:
                    offspring[i, :crossover_point] = parent1[:crossover_point]
                    offspring[i, crossover_point:] = parent2[crossover_point:]

                if i + 1 < self.pop_size:
                    offspring[i + 1, :crossover_point] = parent2[:crossover_point]
                    offspring[i + 1, crossover_point:] = parent1[crossover_point:]
            else:
                # Sin cruce, los hijos son copias de los padres
                if i < self.pop_size:
                    offspring[i] = parent1

                if i + 1 < self.pop_size:
                    offspring[i + 1] = parent2

        return offspring

    def mutate(self, population: torch.Tensor) -> torch.Tensor:
        """
        Aplica mutación a la población
        """
        # Crear máscara de mutación (1 donde ocurrirá mutación, 0 donde no)
        mutation_mask = torch.rand_like(population) < self.mutation_rate

        # No mutar a los individuos élite
        if self.elitism > 0:
            mutation_mask[: self.elitism] = False

        # Aplicar mutación según el tipo de representación
        mutated_population = population.clone()

        if self.binary:
            # Para representación binaria: cambiar 0 por 1 y viceversa
            mutated_population[mutation_mask] = 1 - mutated_population[mutation_mask]
        else:
            # Para representación real: añadir ruido gaussiano
            mutation_noise = torch.randn_like(population) * 0.1
            mutated_population[mutation_mask] += mutation_noise[mutation_mask]
            # Asegurar que los valores están entre 0 y 1
            mutated_population = torch.clamp(mutated_population, 0, 1)

        return mutated_population

    def evolve(self) -> Tuple[torch.Tensor, float]:
        """
        Ejecuta una generación del algoritmo genético

        Returns:
            El mejor cromosoma y su fitness
        """
        # Calcular fitness de la población actual
        fitness = self.calculate_fitness(self.population)

        # Guardar datos para seguimiento
        current_avg_fitness = torch.mean(fitness).item()
        self.avg_fitness_history.append(current_avg_fitness)

        if self.maximize:
            best_idx = torch.argmax(fitness)
        else:
            best_idx = torch.argmin(fitness)

        current_best_fitness = fitness[best_idx].item()
        self.best_fitness_history.append(current_best_fitness)

        if (self.maximize and current_best_fitness > self.best_fitness) or (
            not self.maximize and current_best_fitness < self.best_fitness
        ):
            self.best_fitness = current_best_fitness
            self.best_chromosome = self.population[best_idx].clone()

        # Seleccionar padres
        parents_indices = self.select_parents(fitness)

        # Crear nueva generación mediante cruce
        offspring = self.crossover(parents_indices)

        # Aplicar mutación
        self.population = self.mutate(offspring)

        return self.best_chromosome, self.best_fitness

    def run(self, generations: int, verbose: bool = True) -> Tuple[torch.Tensor, float]:
        """
        Ejecuta el algoritmo genético durante un número determinado de generaciones

        Args:
            generations: Número de generaciones a ejecutar
            verbose: Si es True, imprime información durante la ejecución

        Returns:
            El mejor cromosoma encontrado y su fitness
        """
        for gen in range(generations):
            best_chromosome, best_fitness = self.evolve()

            if verbose and ((gen + 1) % 10 == 0 or gen == 0 or gen == generations - 1):
                print(
                    f"Generación {gen + 1}: Mejor fitness = {best_fitness:.2f}, Fitness promedio = {self.avg_fitness_history[-1]:.2f}"
                )

        if verbose:
            print(f"\nMejor solución encontrada:")
            print(f"Fitness: {self.best_fitness:.2f}")
            if self.binary:
                print(f"Cromosoma: {self.best_chromosome.int().tolist()}")
            else:
                print(f"Cromosoma: {self.best_chromosome.tolist()}")

        return self.best_chromosome, self.best_fitness

    def plot_history(self, title: str = "Evolución del fitness"):
        """
        Grafica el historial de fitness durante la evolución

        Args:
            title: Título del gráfico
        """
        plt.figure(figsize=(10, 5))
        generations = range(1, len(self.best_fitness_history) + 1)

        plt.plot(generations, self.best_fitness_history, "b-", label="Mejor fitness")
        plt.plot(generations, self.avg_fitness_history, "r-", label="Fitness promedio")

        plt.title(title)
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.legend()
        plt.grid(True)
        plt.show()
