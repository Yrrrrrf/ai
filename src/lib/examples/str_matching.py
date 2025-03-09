import torch
import numpy as np
from dataclasses import dataclass
import string

from lib.ga import GeneticAlgorithm

# Configuración para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

@dataclass
class StringMatchingData:
    """Clase para almacenar los datos del problema de coincidencia de cadenas"""
    target_string: str
    charset: str  # Conjunto de caracteres posibles
    char_to_idx: dict  # Mapeo de caracteres a índices
    idx_to_char: dict  # Mapeo de índices a caracteres


def create_string_matching_data(target_string: str, charset: str = None) -> StringMatchingData:
    """
    Crea un objeto de datos para el problema de coincidencia de cadenas
    
    Args:
        target_string: Cadena objetivo a coincidir
        charset: Conjunto de caracteres posibles (si es None, se utilizará un conjunto predeterminado)
        
    Returns:
        Objeto StringMatchingData con los datos del problema
    """
    if charset is None:
        # Charset predeterminado: letras, números y símbolos comunes
        charset = string.ascii_letters + string.digits + string.punctuation + ' '
    
    # Crear mapeos para convertir entre caracteres e índices
    char_to_idx = {char: idx for idx, char in enumerate(charset)}
    idx_to_char = {idx: char for idx, char in enumerate(charset)}
    
    return StringMatchingData(
        target_string=target_string,
        charset=charset,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char
    )


def string_matching_fitness(population: torch.Tensor, problem_data: StringMatchingData) -> torch.Tensor:
    """
    Función de fitness para el problema de coincidencia de cadenas
    
    Args:
        population: Tensor de población [pop_size, chromosome_length]
                   Cada "gen" es un valor entre 0 y 1 que se mapeará al índice de un carácter
        problem_data: Datos del problema
        
    Returns:
        Tensor con los valores de fitness de cada individuo
    """
    pop_size, chromosome_length = population.shape
    charset_size = len(problem_data.charset)
    target_indices = torch.tensor([problem_data.char_to_idx[char] for char in problem_data.target_string])
    
    # Escalar y redondear las entradas continuas a índices discretos de caracteres
    # Multiplicamos por charset_size-1 y redondeamos para obtener índices entre 0 y charset_size-1
    indices = torch.round(population * (charset_size - 1)).long()
    
    # Calcular fitness basado en cuántos caracteres coinciden con la cadena objetivo
    fitness = torch.zeros(pop_size)
    
    for i in range(pop_size):
        # Enfoque de fitness basado en la distancia por carácter
        # Cuanto más cerca esté cada carácter del objetivo, mayor será el fitness
        char_distances = 1.0 - torch.abs(indices[i] - target_indices) / charset_size
        fitness[i] = torch.sum(char_distances)
    
    return fitness


def decode_chromosome(chromosome: torch.Tensor, problem_data: StringMatchingData) -> str:
    """
    Decodifica un cromosoma en una cadena
    
    Args:
        chromosome: Tensor del cromosoma [chromosome_length]
        problem_data: Datos del problema
        
    Returns:
        Cadena decodificada
    """
    charset_size = len(problem_data.charset)
    indices = torch.round(chromosome * (charset_size - 1)).long().tolist()
    return ''.join(problem_data.idx_to_char[idx] for idx in indices)


def display_solution(chromosome: torch.Tensor, problem_data: StringMatchingData):
    """
    Muestra la solución encontrada para el problema de coincidencia de cadenas
    
    Args:
        chromosome: Cromosoma (solución) encontrado
        problem_data: Datos del problema
    """
    solution_string = decode_chromosome(chromosome, problem_data)
    target_string = problem_data.target_string
    
    print("\nResultado de la coincidencia de cadenas:")
    print(f"Cadena objetivo: '{target_string}'")
    print(f"Mejor solución:  '{solution_string}'")
    
    # Mostrar coincidencias
    matches = sum(1 for a, b in zip(target_string, solution_string) if a == b)
    match_percentage = (matches / len(target_string)) * 100
    
    print(f"\nCaracteres coincidentes: {matches}/{len(target_string)} ({match_percentage:.2f}%)")
    
    # Visualizar coincidencias por carácter
    print("\nCoincidencias por carácter:")
    for i, (target, solution) in enumerate(zip(target_string, solution_string)):
        match_indicator = "✓" if target == solution else "✗"
        print(f"Posición {i}: Objetivo='{target}' Solución='{solution}' {match_indicator}")


def solve_string_matching():
    # Configuración del problema
    # target_string = "Hello, Genetic Algorithm!"
    target_string = "..............................................................................................."
    population_size = 100
    n_generations = 500
    
    print(f"\033[92mString Matching Problem\033[0m\n")
    print(f"Target string: '{target_string}'")
    
    # Crear objeto de datos para el problema
    problem_data = create_string_matching_data(target_string)
    
    # Crear y ejecutar el algoritmo genético
    ga = GeneticAlgorithm(
        pop_size=population_size,
        chromosome_length=len(target_string),
        fitness_function=string_matching_fitness,
        problem_data=problem_data,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elitism=5,
        maximize=True,
        binary=False  # Usamos representación continua para este problema
    )

    # Seguimiento de las mejores soluciones a lo largo de las generaciones
    best_solutions = []
    
    # Función de callback para mostrar progreso
    def generation_callback(generation, best_chromosome, best_fitness):
        if generation % 10 == 0 or generation == n_generations - 1:
            solution = decode_chromosome(best_chromosome, problem_data)
            best_solutions.append(solution)
            print(f"Generación {generation+1}: '{solution}' (Fitness: {best_fitness:.2f})")
    
    # Ejecutar algoritmo genético con seguimiento
    for gen in range(n_generations):
        best_chromosome, best_fitness = ga.evolve()
        generation_callback(gen, best_chromosome, best_fitness)
    
    # Visualizar resultados
    ga.plot_history(title='Evolución del fitness para el problema de coincidencia de cadenas')
    
    # Mostrar detalles de la solución
    display_solution(ga.best_chromosome, problem_data)
    
    # Visualizar la evolución de las soluciones
    print("\nEvolución de las mejores soluciones:")
    for i, solution in enumerate(best_solutions):
        if i % 5 == 0:  # Mostrar cada 5 generaciones para no saturar la salida
            gen_num = i * 10  # Como guardamos cada 10 generaciones
            print(f"Generación {gen_num+1}: '{solution}'")
