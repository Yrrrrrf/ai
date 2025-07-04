from code_analyzer import code_analyzer_mcp


def main():
    print("\033[H\033[J", end="")
    print("Starting MCP server...")
    code_analyzer_mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
