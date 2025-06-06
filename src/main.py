# src/main.py

from lib.benchmarks.bench_pso import test_all_benchmarks, compare_parameters
from lib.tools.formatting import MenuItems, run_menu_dispatcher


if __name__ == "__main__":
    script_menu_items: MenuItems = {
        "1": ("Run all PSO benchmarks", test_all_benchmarks),
        "2": ("Compare PSO parameters", compare_parameters),
        # Q: Quit option...
        "q": ("Quit Application", None),  # None indicates this is a quit option
    }

    run_menu_dispatcher(title="AI Project Main Menu", menu_items=script_menu_items)
