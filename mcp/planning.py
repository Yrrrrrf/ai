# FILE: mcp/planning.py

"""
Strategic Planning MCP

This server provides master prompts for high-level strategic tasks, such as
architecting new projects and defining a development roadmap.
"""

from mcp.server.fastmcp import FastMCP
from lib.prompt_loader import prompt_loader

# --- Server Definition ---

planning_server = FastMCP(
    name="Strategic Planning",
    instructions="Provides prompts for architecting new projects and creating development blueprints.",
)

# --- Prompt Definitions ---


@planning_server.prompt(title="Get Project Blueprint Guide")
def get_guide_blueprint() -> str:
    """
    Loads the master prompt that guides the user and AI through the process of
    creating a comprehensive GUIDE.md file for a new project.
    """
    return prompt_loader.load("guide-blueprint-skeleton.md")


# --- Server Execution ---


def main():
    """Main function to start the MCP server."""
    print("\033[H\033[J", end="")
    print("🚀 Starting Strategic Planning MCP server...")
    print("✅ Server is ready. It provides tools for high-level project planning.")
    planning_server.run()


if __name__ == "__main__":
    main()
