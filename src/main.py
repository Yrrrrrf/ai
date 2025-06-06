import matplotlib.pyplot as plt
from lib.pso import PSO
from lib.bench.pso import benchmark_functions, run_benchmark


def test_all_benchmarks():
    """Run PSO on all benchmark functions in 2D"""
    results = {}

    # Common parameters
    dimensions = 2
    num_particles = 30
    max_iterations = 100

    # Create a figure for the plots
    plt.figure(figsize=(15, 10))

    # Test each function
    for i, (name, info) in enumerate(benchmark_functions.items()):
        print(f"\n{'=' * 50}")

        # Run optimization
        best_position, best_fitness, pso = run_benchmark(
            function_name=name,
            pso_class=PSO,
            dimensions=dimensions,
            num_particles=num_particles,
            max_iterations=max_iterations,
            w=0.5,  # Inertia weight
            c1=1.5,  # Cognitive weight
            c2=1.5,  # Social weight
        )

        # Store results
        results[name] = {
            "position": best_position,
            "fitness": best_fitness,
            "history": pso.best_fitness_history.copy(),
        }

        # Plot the convergence history
        plt.subplot(2, 4, i + 1)
        plt.plot(range(1, len(pso.best_fitness_history) + 1), pso.best_fitness_history)
        plt.title(f"{name.capitalize()}")
        plt.xlabel("Iteration")
        plt.ylabel("Best Fitness")
        plt.grid(True)

        # Visualize particles for 2D functions (optional)
        if dimensions == 2:
            pso.visualize_particles_2d(interval=100)

    plt.tight_layout()
    plt.savefig("pso_benchmarks_convergence.png")
    plt.show()

    return results


def compare_parameters():
    """Compare different parameter settings on a chosen function"""
    # Function to test
    function_name = "ackley"
    dimensions = 2
    num_particles = 30
    max_iterations = 100

    # Parameter combinations to test
    parameter_sets = [
        {"w": 0.5, "c1": 1.5, "c2": 1.5, "label": "Default"},
        {"w": 0.7, "c1": 1.0, "c2": 2.0, "label": "Higher social"},
        {"w": 0.7, "c1": 2.0, "c2": 1.0, "label": "Higher cognitive"},
        {"w": 0.9, "c1": 1.5, "c2": 1.5, "label": "Higher inertia"},
        {"w": 0.3, "c1": 1.5, "c2": 1.5, "label": "Lower inertia"},
    ]

    # Create a figure for comparison
    plt.figure(figsize=(10, 6))

    # Run with each parameter set
    for params in parameter_sets:
        label = params.pop("label")

        print(f"\n{'=' * 50}")
        print(f"Testing {function_name} with {label} parameters:")

        # Run optimization
        _, _, pso = run_benchmark(
            function_name=function_name,
            pso_class=PSO,
            dimensions=dimensions,
            num_particles=num_particles,
            max_iterations=max_iterations,
            **params,
        )

        # Plot the convergence history
        plt.plot(
            range(1, len(pso.best_fitness_history) + 1),
            pso.best_fitness_history,
            label=f"{label} (w={params['w']}, c1={params['c1']}, c2={params['c2']})",
        )

    plt.title(f"Parameter Comparison on {function_name.capitalize()} Function")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.legend()
    plt.yscale("log")  # Log scale often helps visualize convergence better
    plt.savefig("pso_parameter_comparison.png")
    plt.show()


def main():
    print("\033[92mPSO Benchmark Suite\033[0m\n")

    # Choose which test to run
    choice = input(
        "Select test: \n1. Test all benchmark functions\n2. Compare parameters\nChoice (1/2): "
    )

    if choice == "1":
        test_all_benchmarks()
    elif choice == "2":
        compare_parameters()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
