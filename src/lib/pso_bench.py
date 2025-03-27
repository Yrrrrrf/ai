import numpy as np

# Collection of benchmark functions for optimization algorithms

def sphere(x):
    """
    Sphere function - the simplest unimodal test function.
    Global minimum at x = 0 where f(x) = 0.
    Bounds: [-5.12, 5.12] for all dimensions
    """
    return np.sum(x**2)

def rosenbrock(x):
    """
    Rosenbrock function - a challenging unimodal function with a narrow valley.
    Global minimum at x = [1, 1, ..., 1] where f(x) = 0.
    Bounds: [-5, 10] for all dimensions
    """
    result = 0
    for i in range(len(x) - 1):
        result += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
    return result

def ackley(x):
    """
    Ackley function - has many local minima but one global minimum.
    Global minimum at x = 0 where f(x) = 0.
    Bounds: [-32.768, 32.768] for all dimensions
    """
    a, b, c = 20, 0.2, 2*np.pi
    n = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + a + np.exp(1)

def griewank(x):
    """
    Griewank function - many local minima with regular distribution.
    Global minimum at x = 0 where f(x) = 0.
    Bounds: [-600, 600] for all dimensions
    """
    sum_term = np.sum(x**2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
    return 1 + sum_term - prod_term

def schwefel(x):
    """
    Schwefel function - many local minima, global minimum far from origin.
    Global minimum at x = [420.9687, 420.9687, ...] where f(x) = 0.
    Bounds: [-500, 500] for all dimensions
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

def himmelblau(x):
    """
    Himmelblau's function - has four identical local minima.
    Only works in 2D. Global minima at:
    (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    Bounds: [-5, 5] for both dimensions
    """
    if len(x) != 2:
        raise ValueError("Himmelblau's function is only defined for 2D problems")
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def easom(x):
    """
    Easom function - has a global minimum in a small area.
    Only works in 2D. Global minimum at (π, π) where f(x) = -1.
    Bounds: [-100, 100] for both dimensions
    """
    if len(x) != 2:
        raise ValueError("Easom function is only defined for 2D problems")
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0]-np.pi)**2 + (x[1]-np.pi)**2))

# Dictionary of functions with their standard bounds
benchmark_functions = {
    'sphere': {
        'function': sphere,
        'bounds': (-5.12, 5.12),
        'description': 'Sphere function - simplest unimodal test'
    },
    'rosenbrock': {
        'function': rosenbrock,
        'bounds': (-5, 10),
        'description': 'Rosenbrock function - challenging valley optimization'
    },
    'ackley': {
        'function': ackley,
        'bounds': (-32.768, 32.768),
        'description': 'Ackley function - many local minima'
    },
    'griewank': {
        'function': griewank,
        'bounds': (-600, 600),
        'description': 'Griewank function - many regularly distributed local minima'
    },
    'schwefel': {
        'function': schwefel,
        'bounds': (-500, 500),
        'description': 'Schwefel function - deceptive global optimum'
    },
    'himmelblau': {
        'function': himmelblau,
        'bounds': (-5, 5),
        'description': 'Himmelblau function - four identical local minima (2D only)'
    },
    'easom': {
        'function': easom,
        'bounds': (-100, 100),
        'description': 'Easom function - small area global minimum (2D only)'
    },
    'rastrigin': {
        'function': lambda x: 10 * len(x) + sum([(xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x]),
        'bounds': (-5.12, 5.12),
        'description': 'Rastrigin function - highly multimodal with regular local minima'
    }
}

def run_benchmark(function_name, pso_class, dimensions=2, num_particles=30, max_iterations=100, **kwargs):
    """
    Run a benchmark test on the specified function using PSO.
    
    Args:
        function_name: Name of the function to optimize (must be in benchmark_functions)
        pso_class: The PSO class to use
        dimensions: Number of dimensions for the problem
        num_particles: Number of particles to use
        max_iterations: Maximum number of iterations
        **kwargs: Additional arguments to pass to the PSO constructor
        
    Returns:
        best_position, best_fitness, pso_instance
    """
    if function_name not in benchmark_functions:
        raise ValueError(f"Unknown function: {function_name}. Available functions: {list(benchmark_functions.keys())}")
    
    func_info = benchmark_functions[function_name]
    function = func_info['function']
    bounds = func_info['bounds']
    
    # Check for 2D-only functions
    if function_name in ['himmelblau', 'easom'] and dimensions != 2:
        print(f"Warning: {function_name} is only defined for 2D problems. Setting dimensions to 2.")
        dimensions = 2
    
    print(f"\n\033[92mRunning PSO on {function_name.capitalize()} Function\033[0m")
    print(f"Dimensions: {dimensions}, Particles: {num_particles}, Max Iterations: {max_iterations}")
    print(f"Description: {func_info['description']}")
    print(f"Bounds: {bounds}")
    
    # Initialize PSO
    pso = pso_class(
        num_particles=num_particles,
        dimensions=dimensions,
        fitness_function=function,
        bounds=bounds,
        max_iterations=max_iterations,
        minimize=True,
        **kwargs
    )
    
    # Run optimization
    best_position, best_fitness = pso.optimize()
    
    # Print results
    print("\nOptimization Results:")
    print(f"Best position: {best_position}")
    print(f"Best fitness: {best_fitness:.6f}")
    
    return best_position, best_fitness, pso