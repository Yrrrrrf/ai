# src/main.py

from lib.benchmarks.bench_pso import test_all_benchmarks, compare_parameters
from lib.tools.formatting import *  # import formatting utilities


def main_dispatcher():
    print_header("Main script")

    menu_items = {
        "1": ("Run all PSO benchmarks", test_all_benchmarks),
        "2": ("Compare PSO parameters", compare_parameters),
        # Add more options here if needed
        "q": ("Quit", None),
    }

    while True:
        print(f"{TermStyle.BOLD}Available Actions:{TermStyle.ENDC}")
        for key, (description, _) in menu_items.items():
            print_menu_item(key, description)

        choice = (
            input(f"\n{TermStyle.BOLD}➡️  Enter your choice: {TermStyle.ENDC}")
            .strip()
            .lower()
        )

        if choice in menu_items:
            description, action_func = menu_items[choice]
            if action_func:
                print_running_message(description)
                try:
                    action_func()  # Call the selected function
                except Exception as e:
                    print_error_message(
                        f"An error occurred during '{description}': {e}"
                    )
                print_finished_message(description)
            elif choice == "q":
                print_exit_message()
                break
        else:
            print_error_message("Invalid choice. Please try again.")
        print(
            f"{TermStyle.DIM}-------------------------------------------{TermStyle.ENDC}"
        )


if __name__ == "__main__":
    main_dispatcher()
