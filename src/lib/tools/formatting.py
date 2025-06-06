# ANSI escape codes for colors and styles
class TermStyle:
    HEADER = "\033[95m"  # Light Magenta
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"  # Reset to default
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"  # For subtle text


def print_header(title):
    """Prints a styled header."""
    print(
        f"\n{TermStyle.BOLD}{TermStyle.HEADER}╔{'═' * (len(title) + 2)}╗{TermStyle.ENDC}"
    )
    print(f"{TermStyle.BOLD}{TermStyle.HEADER}║ {title} ║{TermStyle.ENDC}")
    print(
        f"{TermStyle.BOLD}{TermStyle.HEADER}╚{'═' * (len(title) + 2)}╝{TermStyle.ENDC}\n"
    )


def print_menu_item(key, description):
    """Prints a styled menu item."""
    print(
        f"  {TermStyle.OKBLUE}{TermStyle.BOLD}{key}{TermStyle.ENDC}. {TermStyle.OKCYAN}{description}{TermStyle.ENDC}"
    )


def print_running_message(description):
    """Prints a styled message when an action starts."""
    print(
        f"\n{TermStyle.DIM}───────────────────────────────────────────{TermStyle.ENDC}"
    )
    print(
        f"{TermStyle.OKGREEN}{TermStyle.BOLD}🚀 Running: {description}{TermStyle.ENDC}"
    )
    print(f"{TermStyle.DIM}───────────────────────────────────────────{TermStyle.ENDC}")


def print_finished_message(description):
    """Prints a styled message when an action finishes."""
    print(
        f"\n{TermStyle.DIM}───────────────────────────────────────────{TermStyle.ENDC}"
    )
    print(
        f"{TermStyle.OKGREEN}{TermStyle.BOLD}✅ Finished: {description}{TermStyle.ENDC}"
    )
    print(
        f"{TermStyle.DIM}───────────────────────────────────────────{TermStyle.ENDC}\n"
    )


def print_error_message(message):
    """Prints a styled error message."""
    print(f"  {TermStyle.FAIL}{TermStyle.BOLD}❌ Error: {message}{TermStyle.ENDC}")


def print_exit_message():
    """Prints a styled exit message."""
    print(
        f"\n{TermStyle.WARNING}👋 Exiting PSO Benchmark Suite. Goodbye!{TermStyle.ENDC}"
    )


# * Main dispatch function for printing messages
# src/main.py

from typing import Dict, Tuple, Callable, Optional

# Assuming your project structure allows this import when run correctly
# (e.g., `python -m src.main` from project root, or PYTHONPATH is set)
from lib.benchmarks.bench_pso import test_all_benchmarks, compare_parameters
from lib.tools.formatting import *  # Import formatting utilities

# Define the type for menu items for clarity
MenuItemAction = Optional[Callable[[], None]]  # An action is a callable or None
MenuItems = Dict[str, Tuple[str, MenuItemAction]]


def run_menu_dispatcher(
    title: str, menu_items: MenuItems, prompt_message: str = "➡️  Enter your choice: "
) -> None:
    """
    Displays a menu and executes actions based on user input.

    Args:
        title: The title to display at the top of the menu.
        menu_items: A dictionary where keys are choices (e.g., "1", "q")
                    and values are tuples of (description_str, action_function_or_None).
                    If the action_function is None, it's treated as a quit/exit signal for that item.
        prompt_message: The message to display when asking for user input.
    """
    print_header(title)

    while True:
        print(f"\n{TermStyle.BOLD}Available Actions:{TermStyle.ENDC}")
        for key, (description, _) in menu_items.items():
            print_menu_item(key, description)

        choice = (
            input(f"\n{TermStyle.BOLD}{prompt_message}{TermStyle.ENDC}").strip().lower()
        )

        # Use .get() for safer dictionary access
        selected_item = menu_items.get(choice)

        if not selected_item:
            print_error_message("Invalid choice. Please try again.")
        else:
            description, action_func = selected_item
            if action_func:  # If there's a function to call
                print_running_message(description)
                try:
                    action_func()  # Call the selected function
                except Exception as e:
                    print_error_message(
                        f"An error occurred during '{description}': {e}"
                    )
                print_finished_message(description)
            else:  # If action_func is None, it's the designated quit/exit option
                print_exit_message()
                break  # Exit the while loop

        print(
            f"{TermStyle.DIM}{'─' * 43}{TermStyle.ENDC}"  # Consistent separator
        )


if __name__ == "__main__":
    # Define the specific menu for this main script
    # Note: `test_all_benchmarks` and `compare_parameters` are imported at the top.
    script_menu_items: MenuItems = {
        "1": ("Run all PSO benchmarks", test_all_benchmarks),
        "2": ("Compare PSO parameters", compare_parameters),
        # You can add more script-specific menu items here
        "q": ("Quit Application", None),  # None indicates this is a quit option
    }

    run_menu_dispatcher(title="AI Project Main Menu", menu_items=script_menu_items)
