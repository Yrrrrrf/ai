# FILE: scripts/ingest_gh.py
"""
GitHub Profile Ingestor (Final Polished Version)

This script concurrently ingests all public repositories from a GitHub user,
ignoring the user's profile repository (case-insensitively). It displays
the real-time progress with a rich, clean, and interactive console UI.
"""

import argparse
import sys
import asyncio
from pathlib import Path
import httpx
from typing import List, Dict, Any, Tuple

# --- Import Rich Components ---
from rich.console import Console, RenderableType
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    ProgressColumn,
    Task,
)
from rich.panel import Panel
from rich.text import Text

# --- Configuration ---
MAX_CONCURRENT_TASKS = 8


# --- Custom Rich Progress Column ---
class StatusSpinnerColumn(ProgressColumn):
    """A custom progress column that shows a spinner for active tasks
    and a completion icon ('✓' or '✗') for finished tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Encapsulate a standard spinner column to handle the animation
        self.spinner = SpinnerColumn(spinner_name="dots")

    def render(self, task: "Task") -> RenderableType:
        """Render the spinner or a final status icon."""
        # If the task is finished, display the status icon from the custom field.
        if task.finished:
            # Default to a green check if no specific icon is provided.
            icon = task.fields.get("status_icon", "[bold green]✓[/bold green]")
            return Text.from_markup(icon)

        # If the task is not finished, render the spinner.
        return self.spinner.render(task)


async def get_public_repos(username: str, console: Console) -> List[Dict[str, Any]]:
    """
    Asynchronously fetches a list of public repositories for a user.
    """
    api_url = f"https://api.github.com/users/{username}/repos"

    with console.status(
        f"[bold yellow]Fetching repositories for '{username}'..."
    ) as status:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    api_url, params={"sort": "updated", "per_page": 100}
                )
                response.raise_for_status()

            repos = response.json()
            console.log(f"[bold green]✓ Found {len(repos)} public repositories.")
            return repos

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                console.log(f"[bold red]Error: GitHub user '{username}' not found.")
            else:
                console.log(f"[bold red]Error fetching data from GitHub API: {e}")
            sys.exit(1)
        except Exception as e:
            console.log(f"[bold red]An unexpected error occurred: {e}")
            sys.exit(1)


async def run_ingest_subprocess(repo_url: str) -> Tuple[bool, str]:
    """
    Runs the synchronous 'gitintest.py' script in a separate process.

    Returns:
        A tuple of (success_boolean, stderr_output_string).
    """
    script_path = Path(__file__).parent / "gitintest.py"
    command = [sys.executable, str(script_path), repo_url]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    _, stderr = await process.communicate()

    if process.returncode != 0:
        return False, stderr.decode().strip()

    return True, ""


async def main():
    """Main async function to orchestrate the concurrent ingestion process."""
    parser = argparse.ArgumentParser(
        description="Concurrently ingest all public repositories from a GitHub user with a rich UI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "username", help="The GitHub username to fetch repositories from."
    )
    args = parser.parse_args()

    console = Console()

    # --- Step 1: Fetch and Filter Repositories ---
    all_repos = await get_public_repos(args.username, console)

    # Filter out the special profile repository (CASE-INSENSITIVE)
    profile_repo_name_lower = args.username.lower()
    repositories = [
        repo for repo in all_repos if repo["name"].lower() != profile_repo_name_lower
    ]

    if len(repositories) < len(all_repos):
        console.log(
            f"[yellow]Ignoring profile repository: {args.username}/{args.username}"
        )

    if not repositories:
        console.log("[yellow]No repositories to process. Exiting.")
        return

    # --- Step 2: Set up and run the Rich Progress UI with the custom column ---
    progress_columns = (
        TextColumn("  "),  # Indent for a cleaner look
        StatusSpinnerColumn(),  # Our new custom column!
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    )

    failed_repos = []

    with Progress(*progress_columns, console=console) as progress:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

        main_task_description = (
            f"Ingesting repos for [bold magenta]{args.username}[/bold magenta]"
        )
        main_task = progress.add_task(main_task_description, total=len(repositories))

        async def ingest_with_semaphore(repo: Dict[str, Any]):
            repo_name = repo["name"]
            repo_url = repo["clone_url"]

            # Add a task for this specific repo. The StatusSpinnerColumn will show an animation.
            task_id = progress.add_task(
                f"{repo_name}", total=1, start=False, parent=main_task, visible=True
            )

            async with semaphore:
                progress.start_task(task_id)

                success, error_msg = await run_ingest_subprocess(repo_url)

                if success:
                    # On success, update the task and set the 'status_icon' field for our custom column.
                    progress.update(
                        task_id, completed=1, status_icon="[bold green]✓[/bold green]"
                    )
                else:
                    # On failure, set a different 'status_icon' and record the error.
                    progress.update(
                        task_id, completed=1, status_icon="[bold red]✗[/bold red]"
                    )
                    failed_repos.append((repo_name, error_msg))

                progress.update(main_task, advance=1)

        tasks = [ingest_with_semaphore(repo) for repo in repositories]
        await asyncio.gather(*tasks)

    # --- Step 3: Final Summary (unchanged) ---
    console.print()  # Add a newline for spacing
    console.print("[bold green]Ingestion Complete![/bold green]")
    success_count = len(repositories) - len(failed_repos)
    console.print(f"  [green]Successful:[/green] {success_count}")
    console.print(f"  [red]Failed:    [/red] {len(failed_repos)}")

    if failed_repos:
        console.print("\n--- [bold red]Failure Details[/bold red] ---")
        for name, reason in failed_repos:
            console.print(
                Panel(
                    reason,
                    title=f"Error in repository: [bold]{name}[/bold]",
                    border_style="red",
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
