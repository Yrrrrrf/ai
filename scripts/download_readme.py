"""
Downloads a README.md file from a GitHub repository.
"""

import argparse
import sys
import httpx
from pathlib import Path
from typing import Tuple

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a README.md file from a GitHub repository.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("url", help="Repository clone URL")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "Downloads"),
        help="Directory to save the README.md file",
    )
    return parser.parse_args()

def extract_author_project(url: str) -> Tuple[str, str]:
    """
    Extract author and project names from URL.
    """
    path_parts = [part for part in url.replace(".git", "").split("/") if part]
    if len(path_parts) < 2:
        raise ValueError(f"Cannot extract author and project from URL: {url}")
    return path_parts[-2], path_parts[-1]

def download_readme(author: str, project: str, output_dir: Path) -> None:
    """Downloads the README.md file."""
    branches_to_try = ["main", "master"]
    
    for branch in branches_to_try:
        readme_url = f"https://raw.githubusercontent.com/{author}/{project}/{branch}/README.md"
        try:
            with httpx.stream("GET", readme_url, follow_redirects=True) as response:
                if response.status_code == 200:
                    output_path = output_dir / f"{author}-{project}-README.md"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        for chunk in response.iter_bytes():
                            f.write(chunk)
                    # print(f"Successfully downloaded README to: {output_path}")
                    return
                elif response.status_code != 404:
                    response.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"HTTP error downloading from {readme_url}: {e}", file=sys.stderr)
            sys.exit(1)

    print(f"README.md not found in main or master branch for {author}/{project}", file=sys.stderr)
    sys.exit(1)

def main() -> None:
    """Main function to orchestrate the script execution."""
    args = parse_arguments()
    try:
        author, project = extract_author_project(args.url)
        output_dir = Path(args.output_dir)
        download_readme(author, project, output_dir)
    except ValueError as e:
        print(f"Invalid URL format: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
