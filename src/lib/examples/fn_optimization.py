import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Callable

from lib.ga import GeneticAlgorithm

# Configuración para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class FunctionOptimizationData:
    """Clase para almacenar los datos del problema de optimización de funciones"""
    function: Callable  # La función a optimizar
    function_name: str  # Nombre de la función
    dimensions: int  # Número de dimensiones
    bounds: Tuple[float, float]  # Límites inferior y superior para cada dimensión
    is_minimization: bool  # Si es un problema de minimización (True) o maximización (False)
    optimal_value: float = None  # Valor óptimo conocido de la función (si existe)
    optimal_solution: List[float] = None  # Solución óptima conocida (si existe)


def rastrigin_function(x: torch.Tensor) -> torch.Tensor:
    """
    Función de Rastrigin: f(x) = 10n + sum(x_i^2 - 10cos(2πx_i))
    Mínimo global en x = [0, 0, ..., 0] con f(x) = 0
    
    Args:
        x: Tensor de tamaño [population_size, dimensions]
        
    Returns:
        Valores de la función para cada individuo [population_size]
    """
    n = x.shape[1]
    return 10 * n + torch.sum(x**2 - 10 * torch.cos(2 * np.pi * x), dim=1)


def ackley_function(x: torch.Tensor) -> torch.Tensor:
    """
    Función de Ackley: f(x) = -20exp(-0.2sqrt(0.5*sum(x_i^2))) - exp(0.5*sum(cos(2πx_i))) + 20 + e
    Mínimo global en x = [0, 0, ..., 0] con f(x) = 0
    
    Args:
        x: Tensor de tamaño [population_size, dimensions]
        
    Returns:
        Valores de la función para cada individuo [population_size]
    """
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(x**2, dim=1)))
    term2 = -torch.exp(torch.mean(torch.cos(c * x), dim=1))
    
    return term1 + term2 + a + np.e


def rosenbrock_function(x: torch.Tensor) -> torch.Tensor:
    """
    Función de Rosenbrock (banana function): f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
    Mínimo global en x = [1, 1, ..., 1] con f(x) = 0
    
    Args:
        x: Tensor de tamaño [population_size, dimensions]
        
    Returns:
        Valores de la función para cada individuo [population_size]
    """
    pop_size, dims = x.shape
    result = torch.zeros(pop_size)
    
    for i in range(dims - 1):
        result += 100 * (x[:, i + 1] - x[:, i]**2)**2 + (1 - x[:, i])**2
    
    return result


def function_optimization_fitness(population: torch.Tensor, problem_data: FunctionOptimizationData) -> torch.Tensor:
    """
    Función de fitness para el problema de optimización de funciones
    
    Args:
        population: Tensor de población [pop_size, chromosome_length]
        problem_data: Datos del problema
        
    Returns:
        Tensor con los valores de fitness de cada individuo
    """
    # Decodificar los cromosomas a valores en el dominio de la función
    lower_bound, upper_bound = problem_data.bounds
    decoded_chromosomes = population * (upper_bound - lower_bound) + lower_bound
    
    # Calcular los valores de la función para cada individuo
    function_values = problem_data.function(decoded_chromosomes)
    
    # Si es un problema de minimización, convertimos a fitness
    if problem_data.is_minimization:
        # Para minimización, cuanto menor sea el valor de la función, mayor será el fitness
        # Usamos un valor máximo para asegurar que todos los fitness son positivos
        max_value = torch.max(function_values) + 1
        fitness = max_value - function_values
    else:
        # Para maximización, el fitness es directamente el valor de la función
        fitness = function_values
    
    return fitness


def decode_solution(chromosome: torch.Tensor, problem_data: FunctionOptimizationData) -> torch.Tensor:
    """
    Decodifica un cromosoma en una solución para el problema de optimización
    
    Args:
        chromosome: Tensor del cromosoma [chromosome_length]
        problem_data: Datos del problema
        
    Returns:
        Solución decodificada
    """
    lower_bound, upper_bound = problem_data.bounds
    return chromosome * (upper_bound - lower_bound) + lower_bound


def display_solution(chromosome: torch.Tensor, problem_data: FunctionOptimizationData):
    """
    Muestra la solución encontrada para el problema de optimización de funciones
    
    Args:
        chromosome: Cromosoma (solución) encontrado
        problem_data: Datos del problema
    """
    solution = decode_solution(chromosome, problem_data)
    solution_list = solution.tolist()
    
    # Calcular el valor de la función para la solución
    function_value = problem_data.function(solution.unsqueeze(0)).item()
    
    print(f"\nResultados de la optimización de la función {problem_data.function_name}:")
    print(f"{'Mejor solución encontrada:':<30} {solution_list}")
    print(f"{'Valor de la función:':<30} {function_value}")
    
    if problem_data.optimal_value is not None:
        error = abs(function_value - problem_data.optimal_value)
        print(f"{'Valor óptimo conocido:':<30} {problem_data.optimal_value}")
        print(f"{'Error absoluto:':<30} {error}")
    
    if problem_data.optimal_solution is not None:
        distance = torch.norm(solution - torch.tensor(problem_data.optimal_solution)).item()
        print(f"{'Solución óptima conocida:':<30} {problem_data.optimal_solution}")
        print(f"{'Distancia a la solución óptima:':<30} {distance}")


def plot_function_2d(function, bounds, resolution=100, title="Function Plot"):
    """
    Genera un gráfico 3D de una función de 2 variables
    
    Args:
        function: La función a graficar
        bounds: Límites [min, max] para ambas dimensiones
        resolution: Resolución del gráfico
        title: Título del gráfico
    """
    lower_bound, upper_bound = bounds
    x = np.linspace(lower_bound, upper_bound, resolution)
    y = np.linspace(lower_bound, upper_bound, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Crear tensor para evaluar la función
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    Z = function(points).reshape(resolution, resolution).numpy()
    
    # Crear gráfico 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(title)
    
    # Añadir barra de color
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.show()


def solve_fn_optimization():
    # Elegir la función a optimizar
    print(f"\033[92mFunction Optimization Problem\033[0m\n")
    print("1. Rastrigin Function (más difícil)")
    print("2. Ackley Function (dificultad media)")
    print("3. Rosenbrock Function (dificultad media-alta)")
    
    function_choice = 2  # Podemos cambiar esto para probar diferentes funciones
    
    # Configurar el problema según la elección
    if function_choice == 1:
        function = rastrigin_function
        function_name = "Rastrigin"
        bounds = (-5.12, 5.12)
        optimal_solution = [0.0, 0.0]
        optimal_value = 0.0
    elif function_choice == 2:
        function = ackley_function
        function_name = "Ackley"
        bounds = (-5, 5)
        optimal_solution = [0.0, 0.0]
        optimal_value = 0.0
    else:
        function = rosenbrock_function
        function_name = "Rosenbrock"
        bounds = (-2, 2)
        optimal_solution = [1.0, 1.0]
        optimal_value = 0.0
    
    dimensions = 2  # Usamos 2 dimensiones para poder visualizar
    
    print(f"\nOptimizando la función {function_name} en {dimensions} dimensiones.")
    print(f"Rango de búsqueda: [{bounds[0]}, {bounds[1]}] para cada dimensión.")
    print(f"Solución óptima conocida: {optimal_solution} con valor {optimal_value}.")
    
    # Crear objeto de datos para el problema
    problem_data = FunctionOptimizationData(
        function=function,
        function_name=function_name,
        dimensions=dimensions,
        bounds=bounds,
        is_minimization=True,
        optimal_value=optimal_value,
        optimal_solution=optimal_solution
    )
    
    # Visualizar la función
    plot_function_2d(function, bounds, title=f"{function_name} Function")
    
    # Configuración del algoritmo genético
    population_size = 100
    n_generations = 150
    
    # Crear y ejecutar el algoritmo genético
    ga = GeneticAlgorithm(
        pop_size=population_size,
        chromosome_length=dimensions,
        fitness_function=function_optimization_fitness,
        problem_data=problem_data,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism=5,
        maximize=True,  # Siempre True porque convertimos el problema de minimización
        binary=False  # Usamos representación continua para este problema
    )

    # Ejecutar algoritmo genético
    best_chromosome, best_fitness = ga.run(n_generations)
    
    # Visualizar resultados
    ga.plot_history(title=f'Evolución del fitness para la optimización de la función {function_name}')
    
    # Mostrar detalles de la solución
    display_solution(ga.best_chromosome, problem_data)
    
    # Visualizar la función con la solución encontrada
    solution = decode_solution(ga.best_chromosome, problem_data).numpy()
    
    # Crear gráfico 3D con la solución
    lower_bound, upper_bound = bounds
    x = np.linspace(lower_bound, upper_bound, 100)
    y = np.linspace(lower_bound, upper_bound, 100)
    X, Y = np.meshgrid(x, y)
    
    points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    Z = function(points).reshape(100, 100).numpy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    
    # Marcar la solución encontrada
    ax.scatter(solution[0], solution[1], function(torch.tensor([solution])).item(), 
               color='red', s=100, label='Solución encontrada')
    
    # Marcar la solución óptima
    ax.scatter(optimal_solution[0], optimal_solution[1], optimal_value, 
              color='green', s=100, label='Solución óptima')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title(f'Función {function_name} con solución')
    ax.legend()
    
    plt.show()
