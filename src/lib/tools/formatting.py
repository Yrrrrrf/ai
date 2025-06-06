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
