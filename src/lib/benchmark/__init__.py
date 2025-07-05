# FILE: src/lib/benchmark/__init__.py

"""
Benchmark Suite for AI Models

This package is the central hub for running all benchmark tests for the AI
algorithms in this project. It uses a registry pattern to discover and list
available benchmark suites, which can then be run via an interactive command-line menu.

The main `run()` function orchestrates the menu, while the benchmark functions
themselves are either imported from specialized modules (like `bench_pso.py`) or
defined directly within this file for broader applicability (like `compare_hyperparameters`).

To add a new benchmark suite:
1. Create a new module (e.g., `bench_new_algo.py`) or add a function here.
2. Implement a main function for the test (e.g., `run_new_algo_tests`).
3. If in a new file, import it.
4. Add the function to the `AVAILABLE_BENCHMARKS` list with a description.

This package is self-runnable: `python -m src.lib.benchmark`
"""

# --- Core Imports ---
from typing import List, Tuple, Callable
import matplotlib.pyplot as plt

# --- Project-Specific Imports ---
from lib import ui
from .fn import benchmark_library
from .configs import ALGORITHM_CONFIGS

# --- Import Benchmark Functions from other modules ---
# These are specific analyses tied to a single algorithm.
from .bench_pso import run_convergence_analysis, view_live_animation


# ---------------------------------------------------------------------------- #
#         GENERIC HYPERPARAMETER COMPARISON (DEFINED DIRECTLY IN HUB)          #
# ---------------------------------------------------------------------------- #

def _run_comparison_test(algorithm_class, func_obj, parameter_sets):
    """
    A private helper that runs an algorithm with various parameter sets and plots the results.
    """
    plt.figure(figsize=(12, 8))
    for params in parameter_sets:
        run_params = params.copy()
        label_prefix = run_params.pop("label")
        
        ui.print_message(f"Testing with '{label_prefix}' parameters...", style="cyan")
        
        # Dynamically instantiate and run the algorithm
        algo_instance = algorithm_class(
            num_particles=30, dimensions=2, fitness_function=func_obj,
            bounds=func_obj.bounds, max_iterations=100, minimize=True, **run_params,
        )
        algo_instance.optimize(verbose=False)
        
        param_details = ", ".join([f"{k}={v}" for k, v in run_params.items()])
        full_label = f"{label_prefix} ({param_details})"

        plt.plot(range(len(algo_instance.best_fitness_history)), algo_instance.best_fitness_history, label=full_label)

    plt.title(f"{algorithm_class.__name__} Hyperparameter Comparison on {func_obj.name} Function")
    plt.xlabel("Iteration"); plt.ylabel("Best Fitness (log scale)")
    plt.grid(True); plt.legend(); plt.yscale("log"); plt.tight_layout()
    plt.show()


def compare_hyperparameters():
    """
    Interactively allows the user to select a benchmark function to compare
    different hyperparameter settings for an algorithm.
    """
    algorithm_name = "PSO"  # In the future, this could be a user-selected choice
    config = ALGORITHM_CONFIGS[algorithm_name]

    # --- Let the user choose which function to test against ---
    runnable_functions = [f for f in benchmark_library if f.fixed_dims is None or f.fixed_dims == 2]
    
    ui.console.print(f"Select a Function for {algorithm_name} Hyperparameter Comparison")
    menu_rows = [[str(i + 1), f.name] for i, f in enumerate(runnable_functions)]
    ui.display_table("Available 2D Functions", ["Choice", "Function"], menu_rows)
    
    choice = ui.ask_prompt("Enter your choice (or 'q' to cancel)", default="q")
    if choice.lower() == 'q': return

    try:
        index = int(choice) - 1
        if not (0 <= index < len(runnable_functions)): raise IndexError
        func_obj = runnable_functions[index]
    except (ValueError, IndexError):
        ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        return

    # --- Run the comparison using the chosen function ---
    ui.print_rule()
    _run_comparison_test(
        algorithm_class=config["class"],
        func_obj=func_obj,
        parameter_sets=config["parameter_sets"]
    )


# ---------------------------------------------------------------------------- #
#                         BENCHMARK REGISTRY & RUNNER                          #
# ---------------------------------------------------------------------------- #

BenchmarkAction = Callable[[], None]
BenchmarkSuite = Tuple[str, BenchmarkAction]

# This list is the single source of truth for the main menu.
AVAILABLE_BENCHMARKS: List[BenchmarkSuite] = [
    (
        "Run batch convergence analysis for all functions",
        run_convergence_analysis,
    ),
    (
        "Compare PSO hyperparameters on a chosen function",
        compare_hyperparameters,
    ),
    (
        "View a live 2D animation for a chosen function",
        view_live_animation,
    ),
]


def run() -> None:
    """
    Displays an interactive menu and executes the selected benchmark suite.

    This function is the main entry point for the benchmark runner. It dynamically
    builds a menu from the `AVAILABLE_BENCHMARKS` registry and handles user
    input to launch the corresponding test.
    """
    while True:
        menu_rows = [[str(i + 1), desc] for i, (desc, _) in enumerate(AVAILABLE_BENCHMARKS)]
        menu_rows.append(["q", "Quit Application"])

        ui.display_table(
            title="Available Benchmarks",
            columns=["Choice", "Description"],
            rows=menu_rows,
        )

        choice = ui.ask_prompt("Enter your choice", default="q").lower()
        if choice == "q":
            ui.print_message("\n👋 Exiting Benchmark Suite. Goodbye!", style="yellow")
            break

        try:
            index = int(choice) - 1
            if 0 <= index < len(AVAILABLE_BENCHMARKS):
                description, action_func = AVAILABLE_BENCHMARKS[index]
                ui.print_rule(style="green")
                ui.print_message(f"Running: {description}", prefix="🚀 ", style="bold green")
                try:
                    action_func()
                    ui.print_message(f"Finished: {description}", prefix="✅ ", style="bold green")
                except Exception as e:
                    ui.print_message(f"An error occurred during '{description}': {e}", prefix="❌ ", style="bold red")
            else:
                ui.print_message("Invalid choice. Please select a number from the list.", prefix="❌ ", style="bold red")
        except ValueError:
            ui.print_message(f"Invalid input '{choice}'. Please enter a number or 'q'.", prefix="❌ ", style="bold red")

        ui.print_rule()


# This allows the package to be run directly from the command line,
# making it a self-contained, executable component.
# Usage: python -m src.lib.benchmark
if __name__ == "__main__":
    run()