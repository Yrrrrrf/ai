from code_analyzer import code_analyzer_mcp
from test_server import test_mcp


def main():
    print("\033[H\033[J", end="")
    print("Starting MCP server...")
    # * it only handles the 1st server, the code_analyzer_mcp!
    code_analyzer_mcp.run(transport="stdio")
    test_mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
