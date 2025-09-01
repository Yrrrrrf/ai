# FILE: mcp/analysis.py

"""
Code Intelligence MCP

A focused MCP server that provides a suite of prompts for advanced code
analysis, refactoring, explanation, and generation tasks. This server acts as
an "expert prompter" for any task related to understanding or improving code.
"""

from mcp.server.fastmcp import FastMCP
from typing import Optional
from lib.prompt_loader import prompt_loader

# --- Server Definition ---

analysis_server = FastMCP(
    name="Code Intelligence",
    instructions="A server that generates structured prompts for code analysis, refactoring, and documentation tasks.",
)

# --- Prompt Definitions ---


@analysis_server.prompt(title="Analyze Project Codebase")
def analyze_codebase() -> str:
    """
    Generates a master prompt for performing a comprehensive analysis of the entire codebase.
    Instructs the LLM to cover the project's mission, technology stack, features, strengths, and weaknesses.
    """
    return prompt_loader.load("project-analysis.md")


@analysis_server.prompt(title="Get Refactoring Advice")
def get_refactoring_advice(
    goal: str, file_path: str, constraints: Optional[str] = None
) -> str:
    """
    Generates a master prompt for providing detailed refactoring suggestions for a specific file.

    Args:
        goal (str): The primary objective of the refactoring (e.g., 'reduce complexity').
        file_path (str): The relative path to the file that needs refactoring.
        constraints (Optional[str]): Any limitations to respect (e.g., 'do not change the public API').
    """
    base_prompt = prompt_loader.load("refactoring-guide.md")
    context_header = (
        f"**Task Context for LLM:**\n"
        f"- **File to Refactor:** `{file_path}`\n"
        f"- **Primary Goal:** '{goal}'"
    )
    if constraints:
        context_header += f"\n- **Constraints:** '{constraints}'"
    return f"{context_header}\n\n---\n\n{base_prompt}"


@analysis_server.prompt(title="Explain Code")
def explain_code(file_path: str) -> str:
    """
    Generates a master prompt for creating a detailed, step-by-step explanation of a specific file.

    Args:
        file_path (str): The relative path to the file that needs to be explained.
    """
    base_prompt = prompt_loader.load("code-explanation.md")
    context_header = f"**Task Context for LLM:** The user has requested an explanation for the code located in the file: `{file_path}`."
    return f"{context_header}\n\n---\n\n{base_prompt}"


@analysis_server.prompt(title="Generate Utility and Tests")
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
        function_signature (str): The function's name and parameters.
        return_type (str): The expected return type of the function.
    """
    base_prompt = prompt_loader.load("utility-and-tests.md")
    context_header = (
        f"**Task Context for LLM:** Generate a utility function and tests with the following specifications:\n"
        f"- **Purpose:** {purpose}\n"
        f"- **Language:** {language}\n"
        f"- **Testing Framework:** {test_framework}\n"
        f"- **Function Signature:** `{function_signature}`\n"
        f"- **Return Type:** `{return_type}`"
    )
    return f"{context_header}\n\n---\n\n{base_prompt}"


# --- Server Execution ---


def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")
    print("🚀 Starting Code Intelligence MCP server...")
    print(
        "✅ Server is ready. It exposes prompts for code analysis, refactoring, and generation."
    )
    analysis_server.run()


if __name__ == "__main__":
    main()
