# src/main.py
# main entry point for the application...

# it's more like the main testing script!

# * imports
from lib.ui import *
from lib.benchmark import run as run_benchmarks


def main():
    console.clear()
    console.print(Panel("[green]AI Learning Journey[/]", border_style="green dim"))

    run_benchmarks()


if __name__ == "__main__":
    main()
