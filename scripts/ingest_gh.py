# FILE: scripts/ingest_all_from_gh_async.py
"""
GitHub Profile Ingestor (Asynchronous Version)

This script fetches all public repositories for a given GitHub user and
processes each one concurrently using asyncio and separate subprocesses
for significant speed improvement.
"""

import argparse
import sys
import asyncio
from pathlib import Path
import httpx
from typing import List, Dict, Any, Coroutine

# --- Configuration ---
# Controls how many repositories are cloned at the same time.
# Adjust based on your network and CPU. 5-10 is a safe range.
MAX_CONCURRENT_TASKS = 8


async def get_public_repos(username: str) -> List[Dict[str, Any]]:
    """
    Asynchronously fetches a list of public repositories for a user.
    """
    api_url = f"https://api.github.com/users/{username}/repos"
    print(f"Fetching repositories for '{username}' from {api_url}...")

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                api_url, params={"sort": "updated", "per_page": 100}
            )
            response.raise_for_status()

        repos = response.json()
        print(f"✅ Found {len(repos)} public repositories.")
        return repos

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Error: GitHub user '{username}' not found.", file=sys.stderr)
        else:
            print(f"Error fetching data from GitHub API: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


async def run_ingest_subprocess(repo_name: str, repo_url: str) -> bool:
    """
    Runs the synchronous 'gitintest.py' script in a separate process
    for a single repository.

    Returns:
        True on success, False on failure.
    """
    # Important: Construct the path to the script to run
    script_path = Path(__file__).parent / "gitintest.py"
    
    # We use 'sys.executable' to get the path to the current python interpreter
    # which is the one managed by 'uv'. This is more robust than hardcoding 'python'.
    command = [sys.executable, str(script_path), repo_url]

    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()

    if process.returncode != 0:
        print(
            f"❌ FAILED to process {repo_name}. Stderr:\n{stderr.decode()}",
            file=sys.stderr,
        )
        return False
    else:
        # The original script already prints its success message.
        # We can add one here if we want more verbose async-specific logging.
        return True


async def main():
    """Main async function to orchestrate the concurrent ingestion process."""
    parser = argparse.ArgumentParser(
        description="Concurrently ingest all public repositories from a GitHub user.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "username", help="The GitHub username to fetch repositories from."
    )
    args = parser.parse_args()

    repositories = await get_public_repos(args.username)
    if not repositories:
        print("No repositories to process. Exiting.")
        return

    # A semaphore is a concurrency primitive that will limit the number of
    # concurrent tasks to the value of MAX_CONCURRENT_TASKS.
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    tasks: List[Coroutine] = []
    
    # This wrapper function will use the semaphore
    async def ingest_with_semaphore(repo_name, repo_url):
        async with semaphore:
            print(f"Starting ingestion for: {repo_name}...")
            return await run_ingest_subprocess(repo_name, repo_url)

    for repo in repositories:
        task = ingest_with_semaphore(repo["name"], repo["clone_url"])
        tasks.append(task)
    
    print(f"\nStarting concurrent ingestion of {len(tasks)} repositories ({MAX_CONCURRENT_TASKS} at a time)...\n")
    
    # asyncio.gather runs all tasks concurrently and waits for them all to finish.
    results = await asyncio.gather(*tasks)

    # --- Final Summary ---
    success_count = results.count(True)
    fail_count = results.count(False)
    
    print("=" * 50)
    print("Ingestion Complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    # This is how you run the main async function.
    asyncio.run(main())