# FILE: src/lib/benchmarks/bench_pso.py

"""
PSO Benchmark and Visualization Suites

This script provides three distinct modes for analyzing the PSO algorithm:
1.  Batch Analysis: Run PSO on all functions and generate a summary grid of
    convergence plots.
2.  Hyperparameter Comparison: Compare the performance of different PSO
    parameter settings on a single, challenging function.
3.  Interactive Animation: Select a single function from a menu to view its
    live 2D optimization animation.
"""

import matplotlib.pyplot as plt
import numpy as np

from ..model.pso import PSO
from ..tools.asset_manager import AssetManager
from lib import ui
from .fn import benchmark_library


def run_convergence_analysis():
    """
    Runs PSO on all 2D functions in batch mode and generates a single
    grid of convergence plots. This is a non-interactive analysis.
    """
    # --- Configuration ---
    dimensions = 2
    num_particles = 30
    max_iterations = 100
    w, c1, c2 = 0.5, 1.5, 1.5

    # --- Phase 1: Run all optimizations and collect results ---
    ui.print_message(
        "Starting batch optimization for all 2D functions...", style="bold yellow"
    )
    results = []
    runnable_functions = [
        f
        for f in benchmark_library
        if f.fixed_dims is None or f.fixed_dims == dimensions
    ]

    for func_obj in runnable_functions:
        ui.print_rule()
        ui.print_message(f"Running PSO on {func_obj.name} Function", style="bold cyan")
        pso = PSO(
            num_particles=num_particles,
            dimensions=dimensions,
            fitness_function=func_obj,
            bounds=func_obj.bounds,
            max_iterations=max_iterations,
            w=w,
            c1=c1,
            c2=c2,
            minimize=True,
        )
        pso.optimize(verbose=False)
        results.append({"func_obj": func_obj, "pso_instance": pso})
        ui.print_message(
            f"  Finished {func_obj.name}. Best fitness: {pso.global_best_fitness:.6f}",
            style="green",
        )

    ui.print_rule(style="bold yellow")
    ui.print_message(
        "All optimizations complete. Generating final convergence grid...",
        style="bold yellow",
    )

    # --- Phase 2: Create the final summary plot ---
    num_plots = len(results)
    cols = 4
    rows = (num_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("PSO Convergence Analysis", fontsize=16, weight="bold")
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.plot(
            range(len(res["pso_instance"].best_fitness_history)),
            res["pso_instance"].best_fitness_history,
        )
        ax.set_title(res["func_obj"].name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness")
        ax.grid(True)

    for j in range(num_plots, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = AssetManager.get_temp("pso_benchmarks_convergence.png")
    fig.savefig(save_path)
    ui.print_message(f"Saved convergence plot to: {save_path}", style="yellow")
    plt.show()


def view_live_animation():
    """
    Presents a menu to the user to choose a single 2D function
    and displays its live optimization animation.
    """
    # --- Phase 1: User Selection ---
    runnable_functions = [
        f for f in benchmark_library if f.fixed_dims is None or f.fixed_dims == 2
    ]

    ui.console.print("Select a Function to Animate")
    menu_rows = [[str(i + 1), f.name] for i, f in enumerate(runnable_functions)]
    ui.display_table("Available 2D Functions", ["Choice", "Function"], menu_rows)

    choice = ui.ask_prompt("Enter your choice (or 'q' to cancel)", default="q")
    if choice.lower() == "q":
        return

    try:
        index = int(choice) - 1
        if not (0 <= index < len(runnable_functions)):
            raise IndexError
        func_obj = runnable_functions[index]
    except (ValueError, IndexError):
        ui.print_message("Invalid choice.", prefix="❌ ", style="bold red")
        return

    # --- Phase 2: Run and Animate the Chosen Function ---
    ui.print_rule()
    ui.print_message(f"Preparing animation for {func_obj.name}...", style="bold cyan")
    pso = PSO(
        num_particles=50,
        dimensions=2,
        fitness_function=func_obj,
        bounds=func_obj.bounds,
        max_iterations=100,
        minimize=True,
    )
    pso.optimize(verbose=False)

    # The animation is for viewing only and is not saved.
    pso.visualize_particles_2d(interval=50, title=f"Live Animation: {func_obj.name}")
