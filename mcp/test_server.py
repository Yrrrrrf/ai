# mpc/test_server.py

import random
import time
from mcp.server.fastmcp import FastMCP

# 1. Initialize a completely new, simple server instance.
#    We give it a unique name to avoid any potential caching issues.
test_mcp = FastMCP(
    name="Minimal Test Server",
    instructions="A simple server to generate random numbers for debugging.",
)

# 2. Define a very simple tool with no external dependencies.
@test_mcp.tool()
def generate_random_number(max_value: int = 100) -> str:
    """
    Generates a random integer between 1 and the specified max_value.
    """
    # This print will show up in your terminal if you run it manually.
    print(f"INFO: Tool 'generate_random_number' was called with max_value={max_value}.")
    num = random.randint(1, max_value)
    return f"Your random number is: {num}"


# 3. The main entry point to run the server.
def main():
    """Main function to run the MCP server."""
    # These print statements are crucial for debugging.
    # If we don't see them when running manually, the script itself is failing to start.
    print("\033[H\033[J", end="") # Clear screen
    print("🚀 Starting Minimal Test MCP Server via stdio...")
    print(f"Timestamp: {time.time()}")
    
    # This starts the communication with the Gemini CLI.
    test_mcp.run(transport="stdio")


if __name__ == "__main__":
    main()