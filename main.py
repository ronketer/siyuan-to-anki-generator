#!/usr/bin/env python3
'''Autonomous Local Knowledge to Anki Pipeline.

A multi-agent AI system that extracts knowledge from Siyuan Notes
and creates optimized Anki flashcards using AutoGen.

This project demonstrates:
- Multi-agent orchestration with specialized roles
- Tool use for external API integration
- Reflection pattern for quality assurance
- Human-in-the-loop approval workflow
- Local-first, privacy-preserving AI (no cloud LLM dependencies)

Usage:
    python main.py                  # Run pipeline with human review
    python main.py --block ID       # Override target block ID
'''

import argparse
import asyncio
import json
import re
import sys

from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console

from src.anki_pipeline.agents import create_agents, create_model_client, selector_func
from src.anki_pipeline.config import config


def parse_args() -> argparse.Namespace:
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Generate Anki flashcards from Siyuan Notes using AI agents.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--block', '-b', type=str,
        help='Override TARGET_BLOCK_ID from .env',
    )
    return parser.parse_args()


# ANSI color codes
CYAN = '\033[36m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
MAGENTA = '\033[35m'
BLUE = '\033[34m'
RESET = '\033[0m'
DIM = '\033[2m'

AGENT_STYLES = {
    'user': ('[USER]', CYAN),
    'Knowledge_Manager': ('[KNOWLEDGE MANAGER]', YELLOW),
    'Card_Writer': ('[CARD WRITER]', GREEN),
    'Card_Reviewer': ('[CARD REVIEWER]', MAGENTA),
    'Admin': ('[ADMIN]', BLUE),
}


def extract_json_cards(content: str) -> dict | None:
    '''Extract JSON cards from content that may have surrounding text.'''
    # Try to find JSON in markdown code blocks or raw
    patterns = [
        r'```(?:json)?\s*(\{.*?\})\s*```',  # Markdown code block
        r'(\{"cards":\s*\[.*?\]\})',         # Raw JSON
    ]
    for pattern in patterns:
        match = re.search(pattern, content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


def format_markdown(text: str) -> str:
    '''Format markdown text for terminal display.'''
    # Add blank lines around headers
    text = re.sub(r'(^|\n)(#{1,3})\s+(.+)', r'\1\n\2 \3\n', text)
    # Bold headers
    text = re.sub(r'^(#{1,3})\s+(.+)$', rf'{YELLOW}\2{RESET}', text, flags=re.MULTILINE)
    # Format code blocks
    text = re.sub(r'```(\w+)?\n', f'{DIM}', text)
    text = re.sub(r'```', f'{RESET}', text)
    return text


def format_cards_display(cards: list) -> str:
    '''Format cards for clear display.'''
    lines = []
    for i, card in enumerate(cards, 1):
        front = card.get('front', card.get('question', ''))
        back = card.get('back', card.get('answer', ''))
        lines.append(f'  {DIM}Card {i}:{RESET}')
        lines.append(f'    Q: {front}')
        lines.append(f'    A: {GREEN}{back}{RESET}')
    return '\n'.join(lines)


def format_agent_message(source: str, content: str) -> str:
    '''Format a message with agent name header.'''
    name, color = AGENT_STYLES.get(source, (source, RESET))
    header = f'\n{color}{"-"*50}\n{name}\n{"-"*50}{RESET}'

    # Pretty-print cards if Card_Writer
    if source == 'Card_Writer':
        data = extract_json_cards(content)
        if data and 'cards' in data:
            return f'{header}\n{format_cards_display(data["cards"])}'

    return f'{header}\n{content}'


async def main() -> int:
    '''Run the flashcard generation pipeline.'''
    # Parse CLI arguments
    args = parse_args()

    # Override block ID if provided
    if args.block:
        config.TARGET_BLOCK_ID = args.block

    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            print(f'Configuration Error: {error}')
        print('\nPlease check your .env file.')
        return 1

    print(f'''
{CYAN}===================================================
   Autonomous Knowledge to Anki Pipeline
==================================================={RESET}
  Target: {config.TARGET_BLOCK_ID}
  Model:  {config.LLM_MODEL_ID}

{DIM}Workflow: Fetch -> Write -> Review -> [You Approve] -> Save to Anki
When prompted, type: APPROVE / REJECT / or feedback{RESET}
''')

    model_client = create_model_client()
    agents = create_agents(model_client)

    # Pre-fetch content to work around models that struggle with tool calling
    from src.anki_pipeline.tools import fetch_siyuan_notes
    print(f'{DIM}Fetching content from Siyuan...{RESET}')
    content = fetch_siyuan_notes(config.TARGET_BLOCK_ID)

    if 'Error' in content or 'error' in content.lower():
        print(f'{YELLOW}Warning: {content}{RESET}')
        print('Continuing anyway - model will attempt to fetch...')
        prefetched_content = None
    else:
        # Parse and extract just the markdown
        try:
            data = json.loads(content)
            prefetched_content = data.get('kramdown', content)
            print(f'{GREEN}Content fetched successfully ({len(prefetched_content)} chars){RESET}\n')
        except json.JSONDecodeError:
            prefetched_content = content

    team = SelectorGroupChat(
        participants=[
            agents['knowledge_manager'],
            agents['card_writer'],
            agents['card_reviewer'],
            agents['admin'],
        ],
        model_client=model_client,
        selector_func=selector_func,
        termination_condition=(
            TextMentionTermination('TERMINATE') |
            MaxMessageTermination(30)  # Safety limit
        ),
    )

    # Build task with prefetched content if available
    if prefetched_content:
        task = (
            f"Here is the content from Siyuan Notes:\n\n"
            f"---\n{prefetched_content}\n---\n\n"
            f"Create Anki flashcards from this content. "
            f"Card_Writer: draft the cards. Card_Reviewer: review them. "
            f"Admin will approve. Then Knowledge_Manager saves them with push_cards_batch."
        )
    else:
        task = (
            f"Fetch the notes for Siyuan block ID '{config.TARGET_BLOCK_ID}'. "
            'Draft the Anki cards, and once approved, save them.'
        )

    # Run pipeline and capture result
    result = await Console(team.run_stream(task=task))

    # Fallback: If model didn't properly call push_cards_batch, do it manually
    # This handles smaller models that output pseudo-function-calls as text
    cards_saved = False
    final_cards = None

    for msg in reversed(result.messages):
        # Check if push_cards_batch was successfully called
        if hasattr(msg, 'content') and 'Card added' in str(msg.content):
            cards_saved = True
            break
        # Extract final approved cards from Card_Writer
        if hasattr(msg, 'source') and msg.source == 'Card_Writer':
            final_cards = extract_json_cards(str(msg.content))
            if final_cards:
                break

    # If cards weren't saved but we have approved cards, save them now
    admin_approved = any(
        hasattr(m, 'source') and m.source == 'Admin' and 'APPROVE' in str(m.content).upper()
        for m in result.messages
    )

    if admin_approved and not cards_saved and final_cards:
        print(f'\n{YELLOW}Saving cards (fallback)...{RESET}')
        from src.anki_pipeline.tools import push_cards_batch
        save_result = push_cards_batch(json.dumps(final_cards))
        print(f'{GREEN}{save_result}{RESET}')

    print(f'\n{CYAN}Pipeline complete.{RESET}')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
