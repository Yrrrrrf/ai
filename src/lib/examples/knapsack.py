import torch
import numpy as np
from dataclasses import dataclass

from lib.ga import GeneticAlgorithm


# Configuración para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class KnapsackData:
    """Clase para almacenar los datos del problema de la mochila"""
    weights: torch.Tensor
    values: torch.Tensor
    max_weight: float


def knapsack_fitness(population: torch.Tensor, problem_data: KnapsackData) -> torch.Tensor:
    """
    Función de fitness para el problema de la mochila
    
    Args:
        population: Tensor de población [pop_size, chromosome_length]
        problem_data: Datos del problema (pesos, valores, peso máximo)
        
    Returns:
        Tensor con los valores de fitness de cada individuo
    """
    # Calcular el peso total de cada individuo [pop_size]
    total_weights = torch.matmul(population, problem_data.weights)
    
    # Calcular el valor total de cada individuo [pop_size]
    total_values = torch.matmul(population, problem_data.values)
    
    # Si el peso total excede el máximo, el fitness es 0
    fitness = torch.where(total_weights > problem_data.max_weight, 
                         torch.zeros_like(total_values), 
                         total_values)
    
    return fitness


def display_solution(chromosome: torch.Tensor, problem_data: KnapsackData):
    """
    Muestra la solución encontrada para el problema de la mochila
    
    Args:
        chromosome: Cromosoma (solución) encontrado
        problem_data: Datos del problema
    """
    selected_indices = torch.where(chromosome == 1)[0].tolist()
    
    print("\nObjetos seleccionados:")
    print(f"{'Índice':<10}{'Peso':<10}{'Valor':<10}")
    print("-" * 30)
    
    total_weight = 0
    total_value = 0
    
    for idx in selected_indices:
        weight = problem_data.weights[idx].item()
        value = problem_data.values[idx].item()
        print(f"{idx:<10}{weight:<10.1f}{value:<10.1f}")
        total_weight += weight
        total_value += value
    
    print("-" * 30)
    print(f"{'Total:':<10}{total_weight:<10.1f}{total_value:<10.1f}")
    print(f"Capacidad utilizada: {total_weight}/{problem_data.max_weight} ({total_weight/problem_data.max_weight*100:.1f}%)")


def solve_knapsack():
    # Configuración del problema
    population_size = 50
    n_generations = 100
    n_objects = 8
    bag_capacity = 50

    # Generar pesos y valores aleatorios para los objetos
    weights = torch.randint(1, 10, (n_objects,), dtype=torch.float32)
    values = torch.randint(1, 15, (n_objects,), dtype=torch.float32)

    print(f"\033[92mKnapsack Problem\033[0m\n")
    print(f"Weights: {weights.tolist()}")
    print(f"Values: {values.tolist()}")
    print(f"Maximum weight capacity: {bag_capacity}")
    
    # Crear objeto de datos para el problema
    problem_data = KnapsackData(
        weights=weights,
        values=values,
        max_weight=bag_capacity
    )
    
    # Crear y ejecutar el algoritmo genético
    ga = GeneticAlgorithm(
        pop_size=population_size,
        chromosome_length=n_objects,
        fitness_function=knapsack_fitness,
        problem_data=problem_data,
        mutation_rate=0.05,
        crossover_rate=0.8,
        elitism=2,
        maximize=True,
        binary=True
    )

    best_chromosome, best_fitness = ga.run(n_generations)

    # Visualizar resultados
    ga.plot_history(title='Evolución del fitness para el problema de la mochila')
    
    # Mostrar detalles de la solución
    display_solution(best_chromosome, problem_data)
