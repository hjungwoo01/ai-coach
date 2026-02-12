"""
CLI entry point for the AI Coach.

Usage:
    # Interactive chat mode
    python main.py --config configs/example_sport.json

    # Single query mode
    python main.py --config configs/example_sport.json --query "Who wins, X vs Y?"

    # Use a different OpenAI model
    python main.py --config configs/example_sport.json --model gpt-4o-mini
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from config import SportConfig
from coach import AICoach


def interactive_mode(coach: AICoach):
    """Run interactive chat loop."""
    print(f"\n=== AI {coach.config.sport_name.title()} Coach ===")
    print("Powered by formal methods (PAT/PCSP#) + OpenAI")
    print("Type 'quit' to exit, 'reset' to clear conversation.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if user_input.lower() == 'reset':
            coach.reset()
            print("Conversation cleared.\n")
            continue

        try:
            response = coach.chat(user_input)
            print(f"\nCoach: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main():
    parser = argparse.ArgumentParser(description='AI Sports Coach')
    parser.add_argument(
        '--config', required=True,
        help='Path to sport config JSON file'
    )
    parser.add_argument(
        '--query',
        help='Single query (non-interactive mode)'
    )
    parser.add_argument(
        '--model', default='gpt-4o',
        help='OpenAI model name (default: gpt-4o)'
    )
    parser.add_argument(
        '--output-dir', default='./output',
        help='Directory for generated .pcsp files (default: ./output)'
    )
    args = parser.parse_args()

    # Load .env file (looks in current dir and parent dirs)
    load_dotenv()

    # Load sport config
    config = SportConfig.from_json(args.config)

    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: Set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Create coach
    coach = AICoach(
        config=config,
        openai_api_key=api_key,
        model=args.model,
        output_dir=args.output_dir,
    )

    if args.query:
        response = coach.chat(args.query)
        print(response)
    else:
        interactive_mode(coach)


if __name__ == '__main__':
    main()
