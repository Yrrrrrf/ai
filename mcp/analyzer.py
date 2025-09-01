# FILE: mcp/analysis.py

"""
Code Intelligence MCP (analysis)

A focused MCP server that provides a suite of prompts for advanced code
analysis, refactoring, explanation, and generation tasks.

This server's role is to act as an "expert prompter," providing the LLM with
the detailed, structured instructions needed to perform high-quality,
context-aware work on the user's codebase.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import Optional
from functools import lru_cache

# ==============================================================================
# USER ACTION REQUIRED: Create your master prompt files
# ==============================================================================
# For this script to function, you must create the following Markdown files
# in the directory specified by `PROMPTS_VAULT_DIRECTORY` below.
#
# - project-analysis.md
# - refactoring-guide.md
# - utility-and-tests.md
# - code-explanation.md
# ==============================================================================


# --- 1. Simplified Prompt Template Loading ---

# The hardcoded absolute path to your master prompts directory.
PROMPTS_VAULT_DIRECTORY = Path("/home/yrrrrrf/vault/300-Yrrrrrf/prompts/")


@lru_cache(maxsize=None)
def _load_master_prompt(filename: str) -> str:
    """
    A simple, private helper to load a master prompt template from the vault.
    """
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


# --- 2. MCP Server and Prompt Definitions ---

analysis_mcp = FastMCP(
    name="Code Intelligence MCP",
    instructions="A server that generates structured prompts for code analysis, refactoring, and documentation tasks.",
)


@analysis_mcp.prompt(title="Analyze Project Codebase")
def analyze_codebase() -> str:
    """
    Generates a master prompt for performing a comprehensive analysis of the entire codebase
    currently available in the context. The resulting prompt instructs the LLM to cover
    the project's mission, technology stack, features, strengths, and weaknesses.
    """
    return _load_master_prompt("project-analysis.md")


@analysis_mcp.prompt(title="Get Refactoring Advice")
def get_refactoring_advice(
    goal: str, file_path: str, constraints: Optional[str] = None
) -> str:
    """
    Generates a master prompt for providing detailed refactoring suggestions for a specific file.
    Use this when the user wants to improve a specific piece of code.

    Args:
        goal (str): The primary objective of the refactoring (e.g., 'reduce complexity,' 'improve readability').
        file_path (str): The relative path to the file that needs refactoring.
        constraints (Optional[str]): Any limitations to respect (e.g., 'do not change the public API').
    """
    base_prompt = _load_master_prompt("refactoring-guide.md")

    # Prepend a context header to the master prompt for the LLM.
    context_header = (
        f"**Task Context for LLM:**\n"
        f"- **File to Refactor:** `{file_path}`\n"
        f"- **Primary Goal:** '{goal}'"
    )
    if constraints:
        context_header += f"\n- **Constraints:** '{constraints}'"

    return f"{context_header}\n\n---\n\n{base_prompt}"


@analysis_mcp.prompt(title="Explain Code")
def explain_code(file_path: str) -> str:
    """
    Generates a master prompt for creating a detailed, step-by-step explanation of the code within a specific file.
    Use this when a user is trying to understand how a particular component or module works.

    Args:
        file_path (str): The relative path to the file that needs to be explained.
    """
    base_prompt = _load_master_prompt("code-explanation.md")
    context_header = f"**Task Context for LLM:** The user has requested an explanation for the code located in the file: `{file_path}`."
    return f"{context_header}\n\n---\n\n{base_prompt}"


@analysis_mcp.prompt(title="Generate Utility and Tests")
def generate_utility_and_tests(
    purpose: str,
    language: str,
    test_framework: str,
    function_signature: str,
    return_type: str,
) -> str:
    """
    Generates a master prompt for creating a new utility function and a corresponding set of test cases.

    Args:
        purpose (str): A clear description of what the function should accomplish.
        language (str): The programming language to use (e.g., 'Python', 'TypeScript').
        test_framework (str): The testing framework for the generated tests (e.g., 'pytest', 'Vitest').
        function_signature (str): The function's name and parameters (e.g., 'def calculate_discount(price: float, percentage: float):').
        return_type (str): The expected return type of the function (e.g., 'float').
    """
    base_prompt = _load_master_prompt("utility-and-tests.md")
    context_header = (
        f"**Task Context for LLM:** Generate a utility function and tests with the following specifications:\n"
        f"- **Purpose:** {purpose}\n"
        f"- **Language:** {language}\n"
        f"- **Testing Framework:** {test_framework}\n"
        f"- **Function Signature:** `{function_signature}`\n"
        f"- **Return Type:** `{return_type}`"
    )
    return f"{context_header}\n\n---\n\n{base_prompt}"


# --- 3. Server Execution ---


def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")  # Clears the console screen
    print("🚀 Starting Code Intelligence MCP server...")
    print(f"📘 Loading prompt templates from: {PROMPTS_VAULT_DIRECTORY}")
    print(
        "✅ Server is ready. It exposes prompts for code analysis, refactoring, and generation."
    )
    analysis_mcp.run()


if __name__ == "__main__":
    main()
