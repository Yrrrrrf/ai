# FILE: mcp/release_ops.py

"""
Release Operations MCP (release_ops)

A focused MCP server that provides a hybrid suite of prompts and tools for
automating common GitHub tasks: generating commit messages and creating release notes.

This server follows a two-call pattern:
1. Call a `@prompt` to generate creative text content (like a message or notes).
2. Call a `@tool` with that content to perform a system action (like committing or saving a file).
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

# ==============================================================================
# USER ACTION REQUIRED: Create your master prompt files
# ==============================================================================
# For this script to function, you must create the following Markdown files
# in the directory specified by `PROMPTS_VAULT_DIRECTORY` below.
#
# - new-version-commit.md
# - new-version-release.md  (Use placeholders: {REPO_NAME}, {PREVIOUS_VERSION}, {NEW_VERSION})
# ==============================================================================


# --- 1. Simplified Prompt Template Loading ---

# The hardcoded absolute path to your master prompts directory.
PROMPTS_VAULT_DIRECTORY = Path("/home/yrrrrrf/vault/300-Yrrrrrf/prompts/")


@lru_cache(maxsize=None)  # Still cache the file read for efficiency
def _load_master_prompt(filename: str) -> str:
    """
    A simple, private helper to load a master prompt template from the vault.
    Replaces the need for a full PromptManager class for this focused server.
    """

    # Ensure the base directory exists on first call.
    if not PROMPTS_VAULT_DIRECTORY.is_dir():
        raise FileNotFoundError(
            f"FATAL: Master prompts directory not found at: {PROMPTS_VAULT_DIRECTORY}"
        )

    prompt_file = PROMPTS_VAULT_DIRECTORY / filename
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return (
            f"Error: Master prompt template '{filename}' not found "
            f"in '{PROMPTS_VAULT_DIRECTORY}'. Please ensure the file exists."
        )


# --- 2. MCP Server and Hybrid Function Definitions ---

release_ops = FastMCP(
    name="GitHub Workflow MCP",
    instructions="A server providing prompts and tools for common GitHub tasks like committing code and creating release notes.",
)

# ==============================================================================
# TASK 1: Generating and Executing a Commit
# ==============================================================================


@release_ops.prompt(title="Generate Commit Message Content")
def generate_commit_message_content(focus: Optional[str] = None) -> str:
    """
    Generates ONLY the text content for a conventional Git commit message.
    The agent should call this first to get the message, then call the 'stage_and_commit' tool with the result.

    Args:
        focus (Optional[str]): A hint about the changes (e.g., list of changed files, user summary).
    """
    base_prompt = _load_master_prompt("new-version-commit.md")
    if focus:
        return f"{base_prompt}\n\n**Note to LLM:** The user's focus is: '{focus}'. Tailor the message accordingly."
    return base_prompt


@release_ops.tool()
def stage_and_commit(
    commit_title: str, commit_body: str, files_to_add: List[str]
) -> str:
    """
    Constructs and returns the shell command to stage specified files and create a Git commit with the provided title and body.
    This tool PREPARES the command; it does not execute it. The agent is responsible for execution after user confirmation.

    Args:
        commit_title (str): The main commit title (the first line of the message).
        commit_body (str): The detailed body of the commit message.
        files_to_add (List[str]): A list of file paths to stage with 'git add'. Use '.' to add all.
    """
    if not files_to_add:
        return "Error: No files specified to stage for the commit."

    # Sanitize file paths to handle spaces or just use '.' as is.
    safe_files = " ".join([f'"{f}"' for f in files_to_add])

    add_command = f"git add {safe_files}"
    commit_command = f'git commit -m "{commit_title}" -m "{commit_body}"'

    full_command = f"{add_command} && {commit_command}"

    print(f"INFO: Prepared git command: {full_command}")  # For server-side logging
    return full_command


# ==============================================================================
# TASK 2: Generating and Saving Release Notes
# ==============================================================================


@release_ops.prompt(title="Generate Release Notes Content")
def generate_release_notes_content(
    repo_url: str, old_version: str, new_version: str, focus: Optional[str] = None
) -> str:
    """
    Generates ONLY the Markdown content for GitHub release notes.
    The agent should call this to get the content, then call the 'save_release_notes' tool with the result.

    Args:
        repo_url (str): The full URL of the GitHub repository.
        old_version (str): The previous version tag (e.g., 'v0.1.0').
        new_version (str): The new version tag for this release (e.g., 'v0.2.0').
        focus (Optional[str]): A hint about the changes.
    """
    prompt_template = _load_master_prompt("new-version-release.md")
    repo_name = repo_url.split("/")[-1]

    # Your .md file should contain {REPO_NAME}, {PREVIOUS_VERSION}, and {NEW_VERSION}.
    formatted_prompt = prompt_template.format(
        REPO_NAME=repo_name, PREVIOUS_VERSION=old_version, NEW_VERSION=new_version
    )
    if focus:
        return f"{formatted_prompt}\n\n**Note to LLM:** The user's focus is: '{focus}'. Tailor the notes accordingly."
    return formatted_prompt


@release_ops.tool()
def save_release_notes(content: str, output_path: str = "RELEASE-NOTES.md") -> str:
    """
    Saves the provided content to a release notes file. This performs a file write operation.

    Args:
        content (str): The Markdown content to be written to the file.
        output_path (str): The path where the file should be saved. Defaults to 'RELEASE-NOTES.md'.
    """
    try:
        file_path = Path(output_path)
        file_path.write_text(content, encoding="utf-8")
        return f"Successfully saved release notes to '{file_path.resolve()}'."
    except Exception as e:
        return f"Error: Failed to save release notes to '{output_path}'. Reason: {e}"


# --- 3. Server Execution ---


def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")  # Clears the console screen
    print("🚀 Starting GitHub Workflow MCP server...")
    print(f"📘 Loading prompt templates from: {PROMPTS_VAULT_DIRECTORY}")
    print("✅ Server is ready. It exposes prompts and tools for Git/GitHub tasks.")
    release_ops.run()


if __name__ == "__main__":
    main()
