"""
GitIngest Script - Extract and save repository information
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple
from gitingest import ingest


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract repository information using gitingest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "url",
        nargs="*",  # Allow multiple arguments
        help="Repository URL or local path (supports both Unix '/' and Windows '\\' separators)",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\\Users\\fire\\Downloads",
        help="Directory to save the output file",
    )

    args = parser.parse_args()

    # Join multiple url arguments back into a single path (handles unquoted paths with spaces)
    if isinstance(args.url, list):
        if not args.url:
            parser.error("URL/path argument is required")
        args.url = " ".join(args.url)

    return args


def extract_author_project(url: str) -> Tuple[str, str]:
    """
    Extract author and project names from URL/path.

    Args:
        url: Repository URL or local path

    Returns:
        Tuple of (author, project) names
    """
    # Normalize path separators and split
    path_parts = [part for part in url.replace("\\", "/").split("/") if part]

    if len(path_parts) < 2:
        raise ValueError(f"Cannot extract author and project from URL: {url}")

    # For URLs like "https://github.com/author/project" or local paths
    author, project = path_parts[-2], path_parts[-1]

    # Clean up common URL artifacts
    project = project.replace(".git", "")

    return author, project


def process_repository(url: str) -> Tuple[str, str, str]:
    """
    Process repository using gitingest.

    Args:
        url: Repository URL or local path

    Returns:
        Tuple of (summary, tree, content)
    """
    try:
        summary, tree, content = ingest(url)
        return str(summary), str(tree), str(content)
    except Exception as e:
        print(f"Error processing repository: {e}", file=sys.stderr)
        sys.exit(1)


def save_to_file(
    summary: str, tree: str, content: str, author: str, project: str, output_dir: str
) -> Path:
    """
    Save repository information to file.

    Args:
        summary: Repository summary
        tree: Repository tree structure
        content: Repository content
        author: Author name
        project: Project name
        output_dir: Output directory path

    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filename = f"{author}-{project}.txt"
    file_path = output_path / filename

    # Use list comprehension to build content sections
    sections = [("Summary", summary), ("Tree", tree), ("Content", content)]

    try:
        with file_path.open("w", encoding="utf-8") as f:
            content_lines = [
                f"{section_name}:\n{section_content}\n"
                for section_name, section_content in sections
            ]
            f.write("\n".join(content_lines))

        return file_path
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main function to orchestrate the script execution."""
    # Change to script directory to ensure relative imports work
    script_dir = Path(__file__).parent.parent  # Go up one level from scripts/
    original_cwd = Path.cwd()

    try:
        os.chdir(script_dir)

        args = parse_arguments()

        # Extract author and project information
        author, project = extract_author_project(args.url)
        print(f"Processing: {author}/{project}")

        # Process repository
        summary, tree, content = process_repository(args.url)

        # Save to file
        output_file = save_to_file(
            summary, tree, content, author, project, args.output_dir
        )

        print(f"Successfully saved to: {output_file}")

    except ValueError as e:
        print(f"Invalid URL format: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
