from mcp.server.fastmcp import FastMCP
from typing import Optional, List

# Import our newly refactored library function
# This assumes the mpc server is run from the project root.
from typing import Optional, Tuple, List
from gitingest import ingest


def ingest_repository(
    repo_url: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    branch: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Analyzes a given code repository and returns its summary, file tree, and content.

    This function acts as a clean wrapper around the `gitingest` library,
    handling the ingestion process and returning the structured output.

    Args:
        repo_url (str): The URL or local path of the repository to analyze.
        include_patterns (Optional[List[str]], optional): A list of glob patterns to include.
                                                          Defaults to None.
        exclude_patterns (Optional[List[str]], optional): A list of glob patterns to exclude.
                                                          Defaults to None.
        branch (Optional[str], optional): The specific branch to analyze.
                                          Defaults to the repository's default branch.

    Returns:
        Tuple[str, str, str]: A tuple containing the repository summary,
                              the file tree structure, and the concatenated content of the files.

    Raises:
        Exception: Propagates any exception that occurs during the ingestion process.
    """
    try:
        print(f"INFO: Starting ingestion for repository: {repo_url}")
        summary, tree, content = ingest(
            source=repo_url,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            branch=branch,
        )
        print("INFO: Ingestion successful.")
        return summary, tree, content
    except Exception as e:
        print(f"ERROR: Failed to ingest repository {repo_url}. Reason: {e}")
        # Re-raise the exception so the caller (our MCP tool) can handle it gracefully.
        raise


# Initialize a dedicated FastMCP server for this tool
code_analyzer_mcp: FastMCP = FastMCP(
    name="Code Analyzer MCP",
    instructions="A server that provides tools for analyzing and ingesting code repositories.",
)


@code_analyzer_mcp.tool()
def analyze_codebase(
    repo_url: str,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    branch: Optional[str] = None,
) -> str:
    """
    Analyzes a code repository from a URL or local path and returns a structured summary.
    This tool is ideal for providing the LLM with context about a codebase.

    Args:
        repo_url: The full URL (e.g., 'https://github.com/user/repo') or local path to the repository.
        include_patterns: Optional list of glob patterns to specify which files to include (e.g., ['*.py', '*.md']).
        exclude_patterns: Optional list of glob patterns to specify which files to exclude (e.g., ['dist/*', 'node_modules/*']).
        branch: Optional name of the git branch to analyze. If not provided, the default branch is used.
    """
    try:
        summary, tree, content = ingest_repository(
            repo_url=repo_url,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
            branch=branch,
        )

        # Format the output into a single, LLM-friendly markdown string
        formatted_output = f"""
# Repository Analysis for: {repo_url}

## Summary
```
{summary}
```

## File Tree
```
{tree}
```

---
## Content
{content}
"""
        return formatted_output.strip()

    except Exception as e:
        # Return a clean, user-friendly error message if something goes wrong.
        return f"Error: Failed to analyze repository '{repo_url}'.\nReason: {e}"
