# FILE: src/lib/utils/ui.py

"""
UI Utility Toolkit (using rich)

This module provides a set of generic, reusable functions for creating clean and
styled command-line interfaces. It is purely for presentation and contains no
application-specific logic.
"""

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

# Initialize a single console object for the entire application
console = Console()


def print_message(message: str, prefix: str = "", style: str = "green") -> None:
    """Prints a styled message with an optional prefix.

    Args:
        message (str): The main message content.
        prefix (str, optional): An optional prefix (e.g., an emoji). Defaults to "".
        style (str, optional): The rich style for the message. Defaults to "green".
    """
    console.print(f"[{style}]{prefix}{message}[/{style}]")


def print_rule(style: str = "dim") -> None:
    """Prints a horizontal rule to separate sections."""
    console.rule(style=style)


def ask_prompt(prompt_text: str, default: str = None) -> str:
    """Displays a prompt and safely gets user input.

    Args:
        prompt_text (str): The question to ask the user.
        default (str, optional): A default value if the user enters nothing. Defaults to None.

    Returns:
        str: The user's input.
    """
    return Prompt.ask(f"[bold]➡️  {prompt_text}[/bold]", default=default)


def display_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Displays data in a formatted table.

    Args:
        title (str): The title of the table.
        columns (list[str]): A list of column header names.
        rows (list[list[str]]): A list of rows, where each row is a list of strings.
    """
    table = Table(title=title, border_style="blue")
    table.add_column(columns[0], style="bold cyan", justify="right")
    for col in columns[1:]:
        table.add_column(col, style="cyan")

    for row in rows:
        table.add_row(*row)

    console.print(table)
