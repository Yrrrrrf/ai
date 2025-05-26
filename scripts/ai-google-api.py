"""
Google Gemini API Example

Simple script demonstrating how to use Google's Generative AI API.
"""

import os
from google import genai


def generate_with_gemini(prompt, model="gemini-2.0-flash"):
    """Generate text using Gemini model"""
    try:
        # Get API key and initialize client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Error: Missing API key. Set GEMINI_API_KEY environment variable.")

        # Initialize and call API
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model, contents=prompt)

        return response.text
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    examples = [
        "What is your name?",
        "Explain how AI works in a few words",
        "Tell me your name!",
    ]

    print("=== Google Gemini API Demo ===\n")
    for prompt in examples:
        print(f"Prompt: {prompt}")
        response = generate_with_gemini(prompt)
        print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
