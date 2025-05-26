import random
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
main_mcp: FastMCP = FastMCP(
    name="AI MCP",
    instructions="This is just a prototype of usage of the mcp library.",
)


@main_mcp.tool()
def random_and_square() -> str:
    """Return a random integer and its square."""
    n = random.randint(1, 100)
    return f"Random number: {n}\nSquare: {n**2}"


@main_mcp.tool()
def greet_user(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}! Welcome to the AI MCP."
