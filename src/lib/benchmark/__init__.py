# FILE: src/lib/benchmark/__init__.py

"""
Benchmark Suite for AI Models

This package is the central hub for running all benchmark tests for the AI
algorithms in this project. It features a hierarchical menu system that allows
users to first select an algorithm and then choose a specific benchmark to run.

The core of this suite is its algorithm-agnostic runners, which can operate
on any combination of algorithm and problem, provided they conform to the
project's base classes.

This package is self-runnable: `python -m src.lib.benchmark`
"""

# --- Core Imports ---
from typing import List, Tuple, Callable, Dict, Type
import matplotlib.pyplot as plt

# --- Project-Specific Imports ---
from lib import ui
from ..model.base import OptimizableAlgorithm  # Standardized Algorithm Interface
from ..model.pso import PSO
from ..model.ga import GeneticAlgorithm
from .problems import benchmark_problem_library, BenchmarkProblem, ContinuousProblem
from .configs import ALGORITHM_CONFIGS, DEFAULT_ALGO_PARAMS

# =================================================================== #
#                         GENERIC BENCHMARK RUNNERS                   #
# =================================================================== #


def run_convergence_analysis(algorithm_class: Type[OptimizableAlgorithm]):
    """
    Runs a given algorithm on all compatible problems from the library and
    plots a grid of convergence results. This is algorithm-agnostic.
    """
    ui.print_message(
        f"Starting batch analysis for [bold cyan]{algorithm_class.__name__}[/bold cyan]...",
        style="yellow",
    )
    results = []

    # --- Phase 1: Run all optimizations ---
    for prob in benchmark_problem_library:
        try:
            # Dynamically get the configuration this algorithm needs for this problem
            problem_config = prob.get_config_for_algorithm(algorithm_class.__name__)
            default_params = DEFAULT_ALGO_PARAMS.get(algorithm_class.__name__, {})

            ui.print_rule()
            ui.print_message(f"Running on [bold]{prob.name}[/bold]...", style="cyan")

            # Instantiate and run the algorithm
            algo_instance = algorithm_class(**default_params, **problem_config)
            algo_instance.run(verbose=False)  # Standardized run method

            results.append({"problem_name": prob.name, "instance": algo_instance})
            ui.print_message(
                f"  Finished. Best fitness: {algo_instance.best_fitness_history[-1]:.6f}",
                style="green",
            )

        except NotImplementedError:
            # This is the designed way to skip incompatible algorithm/problem pairs
            continue
        except Exception as e:
            ui.print_message(
                f"Error on [bold]{prob.name}[/bold]: {e}", style="bold red"
            )

    # --- Phase 2: Create the final summary plot ---
    if not results:
        ui.print_message(
            "No compatible problems found or all runs failed.", style="bold red"
        )
        return

    num_plots = len(results)
    cols = 3
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        f"{algorithm_class.__name__} Convergence Analysis", fontsize=16, weight="bold"
    )
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.plot(res["instance"].best_fitness_history)
        ax.set_title(res["problem_name"])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness")
        ax.grid(True)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def compare_hyperparameters(algorithm_class: Type[OptimizableAlgorithm]):
    """
    Interactively allows the user to select a problem and compare different
    hyperparameter settings for the chosen algorithm.
    """
    algo_name = algorithm_class.__name__

    # --- Phase 1: Select a compatible problem ---
    compatible_problems = []
    for prob in benchmark_problem_library:
        try:
            prob.get_config_for_algorithm(algo_name)
            compatible_problems.append(prob)
        except NotImplementedError:
            continue

    if not compatible_problems:
        ui.print_message(
            f"No problems in the library are configured for {algo_name}.", style="red"
        )
        return

    ui.console.print(
        f"\n[bold underline]Select a Problem for {algo_name} Hyperparameter Comparison[/]",
        justify="center",
    )
    menu_rows = [[str(i + 1), p.name] for i, p in enumerate(compatible_problems)]
    ui.display_table("Available Problems", ["Choice", "Problem"], menu_rows)

    choice = ui.ask_prompt("Enter your choice (or 'q' to cancel)", default="q")
    if choice.lower() == "q":
        return

    try:
        chosen_problem = compatible_problems[int(choice) - 1]
        base_config = chosen_problem.get_config_for_algorithm(algo_name)
    except (ValueError, IndexError):
        ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        return

    # --- Phase 2: Run Comparison and Plot Results ---
    ui.print_rule()
    plt.figure(figsize=(12, 8))

    hyperparam_sets = ALGORITHM_CONFIGS[algo_name]["parameter_sets"]
    default_params = DEFAULT_ALGO_PARAMS.get(algo_name, {})

    for params in hyperparam_sets:
        run_params = params.copy()
        label_prefix = run_params.pop("label")

        ui.print_message(f"Testing with '{label_prefix}' parameters...", style="cyan")

        # Combine default, base problem, and hyper-parameters
        final_config = {**default_params, **base_config, **run_params}

        algo_instance = algorithm_class(**final_config)
        algo_instance.run(verbose=False)

        param_details = ", ".join([f"{k}={v}" for k, v in run_params.items()])
        full_label = f"{label_prefix} ({param_details})"

        plt.plot(algo_instance.best_fitness_history, label=full_label)

    plt.title(f"{algo_name} Hyperparameter Comparison on {chosen_problem.name}")
    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# =================================================================== #
#                   ALGORITHM-SPECIFIC VISUALIZATIONS                 #
# =================================================================== #


# todo: See if this fn can fit somewhere else, maybe in a more generic visualization module...
def view_pso_live_animation():
    """
    Presents a menu to the user to choose a single 2D function
    and displays its live optimization animation for PSO.
    """
    # Filter for problems tagged as '2d' and compatible with PSO
    runnable_problems = [
        p
        for p in benchmark_problem_library
        if "2d" in p.tags and isinstance(p, ContinuousProblem)
    ]

    ui.console.print("Select a 2D Function to Animate with PSO")
    menu_rows = [[str(i + 1), p.name] for i, p in enumerate(runnable_problems)]
    ui.display_table("Available 2D Functions", ["Choice", "Function"], menu_rows)

    choice = ui.ask_prompt("Enter your choice (or 'q' to cancel)", default="q")
    if choice.lower() == "q":
        return

    try:
        problem = runnable_problems[int(choice) - 1]
        pso_config = problem.get_config_for_algorithm("PSO")
        default_params = DEFAULT_ALGO_PARAMS["PSO"]
    except (ValueError, IndexError):
        ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        return

    ui.print_rule()
    ui.print_message(f"Preparing animation for {problem.name}...", style="bold cyan")
    pso = PSO(**default_params, **pso_config)
    pso.run(verbose=False)
    pso.visualize_particles_2d(interval=50, title=f"Live Animation: {problem.name}")


# =================================================================== #
#                        BENCHMARK REGISTRY & RUNNER                  #
# =================================================================== #

BenchmarkAction = Callable[[], None]
BenchmarkSuite = Tuple[str, BenchmarkAction]

# The new hierarchical registry. Maps algorithm names to their list of benchmarks.
BENCHMARK_SUITES: Dict[str, List[BenchmarkSuite]] = {
    "PSO": [
        (
            "Run batch convergence analysis on all problems",
            lambda: run_convergence_analysis(PSO),
        ),
        (
            "Compare hyperparameters on a chosen problem",
            lambda: compare_hyperparameters(PSO),
        ),
        ("View a live 2D animation for a chosen function", view_pso_live_animation),
    ],
    "Genetic Algorithm": [
        (
            "Run batch convergence analysis on all problems",
            lambda: run_convergence_analysis(GeneticAlgorithm),
        ),
        (
            "Compare hyperparameters on a chosen problem",
            lambda: compare_hyperparameters(GeneticAlgorithm),
        ),
    ],
}


def _run_algorithm_submenu(algo_name: str, benchmarks: List[BenchmarkSuite]):
    """A helper to run the second level of the menu for a specific algorithm."""
    while True:
        ui.print_rule()
        ui.console.print(f"Benchmarks for: [bold cyan]{algo_name}[/bold cyan]")

        menu_rows = [[str(i + 1), desc] for i, (desc, _) in enumerate(benchmarks)]
        menu_rows.append(["b", "Back to main menu"])
        ui.display_table("Available Tests", ["Choice", "Description"], menu_rows)

        choice = ui.ask_prompt("Enter your choice", default="b").lower()
        if choice == "b":
            break

        try:
            index = int(choice) - 1
            if 0 <= index < len(benchmarks):
                description, action_func = benchmarks[index]
                ui.print_rule(style="green")
                ui.print_message(
                    f"Running: {description}", prefix="🚀 ", style="bold green"
                )
                action_func()
                ui.print_message(
                    f"Finished: {description}", prefix="✅ ", style="bold green"
                )
            else:
                ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        except Exception as e:
            ui.print_message(f"An error occurred: {e}", prefix="❌ ", style="bold red")


def run() -> None:
    """Displays the main interactive menu for selecting an algorithm and its benchmarks."""
    while True:
        ui.console.rule("AI Algorithm Benchmark Suite")

        algorithms = list(BENCHMARK_SUITES.keys())
        menu_rows = [[str(i + 1), name] for i, name in enumerate(algorithms)]
        menu_rows.append(["q", "Quit Application"])
        ui.display_table(
            "Select an Algorithm to Test", ["Choice", "Algorithm"], menu_rows
        )

        choice = ui.ask_prompt("Enter your choice", default="q").lower()
        if choice == "q":
            ui.print_message("\n👋 Goodbye!", style="yellow")
            break

        try:
            index = int(choice) - 1
            if 0 <= index < len(algorithms):
                algo_name = algorithms[index]
                _run_algorithm_submenu(algo_name, BENCHMARK_SUITES[algo_name])
            else:
                ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        except ValueError:
            ui.print_message("Invalid input.", prefix="❌ ", style="bold red")


# This allows the package to be run directly from the command line
if __name__ == "__main__":
    run()
