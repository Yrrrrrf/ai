# FILE: mcp/ai_directives.py

"""
AI Directives MCP (ai_directives)

This server provides a suite of master prompts designed to act as a persistent
system context or behavioral framework for an AI assistant.

Its purpose is to configure the AI's personality, coding style, and strategic
approach at the beginning of a work session, ensuring consistent and high-quality
collaboration.
"""

from mcp.server.fastmcp import FastMCP
from pathlib import Path
from functools import lru_cache

# ==============================================================================
# USER ACTION REQUIRED: Create your master prompt files
# ==============================================================================
# For this script to function, you must create the following Markdown files
# in the directory specified by `PROMPTS_VAULT_DIRECTORY` below.
#
# - core-philosophy.md
# - proactive-advisor.md
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

ai_directives = FastMCP(
    name="AI Directives MCP",
    instructions="A server providing behavioral and stylistic directives to configure an AI assistant for a work session.",
)

# ==============================================================================
# DIRECTIVE 1: The Core Coding Philosophy
# ==============================================================================


@ai_directives.prompt(title="Set Core Development Philosophy")
def get_core_philosophy() -> str:
    """
    Provides the baseline AI directive for coding style. This prompt should be used
    at the start of a session to ensure the AI adheres to modern syntax, functional
    principles, and strict typing. It is language-agnostic.
    """
    return _load_master_prompt("core-philosophy.md")


# ==============================================================================
# DIRECTIVE 2: The Proactive Advisor Mode
# ==============================================================================


@ai_directives.prompt(title="Activate Proactive Advisor Mode")
def get_proactive_advisor_directive() -> str:
    """
    Appends a directive that instructs the AI to act as a strategic partner.
    After fulfilling the primary request, the AI will provide a distinct section
    with suggestions for technical improvements, architectural patterns, and feature ideas.
    """
    return _load_master_prompt("proactive-advisor.md")


# --- 3. Server Execution ---


def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")  # Clears the console screen
    print("🚀 Starting AI Directives MCP server...")
    print(f"📘 Loading prompt templates from: {PROMPTS_VAULT_DIRECTORY}")
    print(
        "✅ Server is ready. It exposes prompts to configure your AI assistant's behavior."
    )
    ai_directives.run()


if __name__ == "__main__":
    main()
