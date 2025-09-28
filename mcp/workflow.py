# FILE: mcp/workflow.py

"""
Git Workflow MCP

A focused MCP server that provides a hybrid suite of prompts and tools for
automating common Git tasks: generating commit messages and creating release notes.
"""

from datetime import datetime
import subprocess
import tempfile
from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional
from lib.prompt_loader import prompt_loader

# --- Server Definition ---

workflow_server = FastMCP(
    name="Git Workflow",
    instructions="A server providing prompts and tools for common Git tasks like committing code and creating release notes.",
)

# --- Prompt Definitions ---


@workflow_server.prompt(title="Generate and Execute a Commit")
def autocommit(focus: Optional[str] = None) -> str:
    """
    Analyzes all current changes (staging them if necessary), generates a
    conventional commit message, and executes it with the `gen_commit_message` tool.
    This is the primary entry point for the autocommit workflow.

    Args:
        focus (Optional[str]): A hint about the changes to help the LLM.
    """
    # This updated prompt now handles the "nothing staged" edge case.
    base_prompt = """
You are an expert software engineer. Your task is to intelligently create and execute a conventional Git commit.

**Your process MUST be as follows:**

1.  **Stage All Changes:** First, you MUST run `git add .` to ensure that all modified and new files are staged. This guarantees that the commit will be comprehensive.

2.  **Analyze Staged Changes:** After staging, you MUST review the context of the staged code changes by inspecting `git diff --staged`.

3.  **Generate a Commit Message:** Based on the changes and the user's focus, write a high-quality conventional commit message with a `title` and a `body`.

4.  **Execute the Commit:** You MUST immediately call the `gen_commit_message` tool to finalize the process. Pass the `commit_title` and `commit_body` you just generated as arguments to the tool.

Do not ask for confirmation at any step. Perform this entire sequence of actions directly.
"""
    if focus:
        return f"{base_prompt}\n\n**User's Focus for this commit is:** '{focus}'. Tailor the commit message accordingly."
    return base_prompt


# In mcp/workflow.py

# This function is no longer a user-facing @workflow_server.prompt
# It's now a helper function that our main prompt will use.
def get_release_notes_template(repo_url: str, old_version: str, new_version: str) -> str:
    """
    Loads and formats the release notes master prompt template.
    This is a helper for the main autorelease workflow.
    """
    prompt_template = prompt_loader.load("new-version-release.md")
    repo_name = repo_url.split("/")[-1]

    # Return the formatted master prompt, which will be given to the LLM
    # as its set of instructions for HOW to write the notes.
    return prompt_template.format(
        REPO_NAME=repo_name, PREVIOUS_VERSION=old_version, NEW_VERSION=new_version
    )

# --- Tool Definitions ---


@workflow_server.tool()
def gen_commit_message(commit_title: str, commit_body: str) -> str:
    """
    Stages all current changes and safely commits them with the provided message
    using a temporary file to prevent shell injection issues.
    """
    try:
        # 1. Stage all changes
        stage_command = ["git", "add", "."]
        stage_process = subprocess.run(
            stage_command, capture_output=True, text=True, check=True
        )

    except subprocess.CalledProcessError as e:
        return f"Error: Failed to stage files.\nGit add stderr:\n{e.stderr}"

    try:
        # 2. Create the full commit message content
        full_commit_message = f"{commit_title}\n\n{commit_body}"

        # 3. Use a temporary file to safely pass the message to Git
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(full_commit_message)
            tmp_file_path = tmp_file.name

        # 4. Execute the commit command using the file
        commit_command = ["git", "commit", "-F", tmp_file_path]
        commit_process = subprocess.run(
            commit_command, capture_output=True, text=True, check=True
        )

        # 5. Clean up the temporary file
        Path(tmp_file_path).unlink()

        # 6. Return the successful output from Git
        return f"Commit successful:\n{commit_process.stdout}"

    except subprocess.CalledProcessError as e:
        # If commit fails, return the error from Git
        return f"Error: Git commit failed.\nGit commit stderr:\n{e.stderr}"
    except Exception as e:
        # Catch any other unexpected errors
        return f"An unexpected error occurred: {e}"


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
