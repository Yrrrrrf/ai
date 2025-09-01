# FILE: mcp/workflow.py

"""
Git Workflow MCP

A focused MCP server that provides a hybrid suite of prompts and tools for
automating common Git tasks: generating commit messages and creating release notes.
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
from lib.prompt_loader import prompt_loader

# --- Server Definition ---

workflow_server = FastMCP(
    name="Git Workflow",
    instructions="A server providing prompts and tools for common Git tasks like committing code and creating release notes.",
)

# --- Prompt Definitions ---

@workflow_server.prompt(title="Generate Commit Message")
def gen_commit_message(focus: Optional[str] = None) -> str:
    """
    Generates the text content for a conventional Git commit message based on staged changes.

    Args:
        focus (Optional[str]): A hint about the changes to help the LLM.
    """
    base_prompt = prompt_loader.load("new-version-commit.md")
    if focus:
        return f"{base_prompt}\n\n**Note to LLM:** The user's focus is: '{focus}'. Tailor the message accordingly."
    return base_prompt


@workflow_server.prompt(title="Generate Release Notes")
def gen_release_notes(
    repo_url: str, old_version: str, new_version: str, focus: Optional[str] = None
) -> str:
    """
    Generates the Markdown content for GitHub release notes.

    Args:
        repo_url (str): The full URL of the repository.
        old_version (str): The previous version tag (e.g., 'v0.1.0').
        new_version (str): The new version tag for this release (e.g., 'v0.2.0').
        focus (Optional[str]): A hint about the changes.
    """
    prompt_template = prompt_loader.load("new-version-release.md")
    repo_name = repo_url.split("/")[-1]

    formatted_prompt = prompt_template.format(
        REPO_NAME=repo_name, PREVIOUS_VERSION=old_version, NEW_VERSION=new_version
    )
    if focus:
        return f"{formatted_prompt}\n\n**Note to LLM:** The user's focus is: '{focus}'. Tailor the notes accordingly."
    return formatted_prompt


# --- Tool Definitions ---

@workflow_server.tool()
def stage_and_commit(
    commit_title: str, commit_body: str, files_to_add: List[str]
) -> str:
    """
    Constructs the shell command to stage files and create a Git commit. Does not execute.

    Args:
        commit_title (str): The main commit title.
        commit_body (str): The detailed body of the commit message.
        files_to_add (List[str]): List of file paths to stage. Use '.' to add all.
    """
    if not files_to_add:
        return "Error: No files specified to stage for the commit."

    safe_files = " ".join([f'"{f}"' for f in files_to_add])
    add_command = f"git add {safe_files}"
    commit_command = f'git commit -m "{commit_title}" -m "{commit_body}"'
    full_command = f"{add_command} && {commit_command}"
    return full_command


@workflow_server.tool()
def save_release_notes(content: str, output_path: str = "RELEASE-NOTES.md") -> str:
    """
    Saves the provided content to a release notes file.

    Args:
        content (str): The Markdown content to be written to the file.
        output_path (str): The path where the file should be saved.
    """
    try:
        file_path = Path(output_path)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully saved release notes to '{file_path.resolve()}'."
    except Exception as e:
        return f"Error: Failed to save release notes. Reason: {e}"


@workflow_server.tool()
def git_snapshot_now() -> str:
    """
    Creates a timestamped 'snapshot' commit, staging all current changes.
    Use when asked to "snapshot" or "save progress" quickly.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d:%H-%M-%S")
    commit_message = f"snapshot {timestamp}"
    stage_command = "git add ."
    commit_command = f'git commit -m "{commit_message}"'
    full_command = f"{stage_command} && {commit_command}"
    return full_command


# --- Server Execution ---

def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")
    print("🚀 Starting Git Workflow MCP server...")
    print("✅ Server is ready. It exposes prompts and tools for Git tasks.")
    workflow_server.run()


if __name__ == "__main__":
    main()