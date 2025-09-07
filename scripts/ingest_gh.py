# FILE: scripts/ingest_gh.py
"""
GitHub Profile Ingestor

This script fetches all public repositories for a given GitHub user and
processes each one using the logic from the 'gitintest.py' script.
"""

import argparse
import sys
import httpx
from typing import List, Dict, Any

# --- Import the core, reusable functions from your existing script ---
# This is the best practice for reusing code and ensures identical behavior.
from gitintest import (
    process_repository,
    save_to_file,
    extract_author_project,
)


def get_public_repos(username: str) -> List[Dict[str, Any]]:
    """
    Fetches a list of public repositories for a user from the GitHub API.

    Args:
        username: The GitHub username.

    Returns:
        A list of repository data dictionaries.
    """
    api_url = f"https://api.github.com/users/{username}/repos"
    print(f"Fetching repositories for '{username}' from {api_url}...")

    try:
        with httpx.Client() as client:
            response = client.get(api_url, params={"sort": "updated", "per_page": 100})
            response.raise_for_status()  # Raise an exception for bad status codes (404, 500, etc.)

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


def main():
    """Main function to orchestrate the ingestion process."""
    parser = argparse.ArgumentParser(
        description="Ingest all public repositories from a GitHub user.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("username", help="The GitHub username to fetch repositories from.")

    args = parser.parse_args()

    # --- Step 1: Get the list of repositories ---
    repositories = get_public_repos(args.username)
    if not repositories:
        print("No repositories to process. Exiting.")
        return

    # --- Step 2: Define the output directory to match gitintest.py behavior ---
    # This ensures the output is saved in the same default location.
    output_directory = "/home/yrrrrrf/Downloads"
    print(f"Output will be saved to: {output_directory}\n")

    total_repos = len(repositories)
    success_count = 0
    fail_count = 0

    # --- Step 3: Iterate and process each repository ---
    for i, repo in enumerate(repositories):
        repo_name = repo["name"]
        repo_url = repo["clone_url"]
        print(f"--- Processing repository {i + 1}/{total_repos}: {repo_name} ---")
        print(f"URL: {repo_url}")

        try:
            # --- Using the imported functions for identical behavior ---
            author, project = extract_author_project(repo_url)
            summary, tree, content = process_repository(repo_url)
            output_file = save_to_file(
                summary, tree, content, author, project, output_directory
            )
            # --- End of imported logic ---

            print(f"✅ Successfully saved to: {output_file}\n")
            success_count += 1

        except Exception as e:
            print(f"❌ FAILED to process {repo_name}. Reason: {e}\n", file=sys.stderr)
            fail_count += 1

    # --- Final Summary ---
    print("=" * 50)
    print("Ingestion Complete!")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {fail_count}")
    print("=" * 50)


if __name__ == "__main__":
    main()
