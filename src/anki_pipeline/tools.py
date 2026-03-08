"""Tool definitions for AI agents.

Tools are functions that agents can call to interact with external systems.
Each tool is annotated with type hints that AutoGen uses to generate
function schemas for the LLM.
"""

import json
import re
from typing import Annotated

import requests

from .config import config


def _clean_kramdown(kramdown: str) -> str:
    """Remove Siyuan metadata from kramdown for cleaner display."""
    # Remove block IDs like {: id="..." updated="..."}
    cleaned = re.sub(r'\{:\s*id="[^"]+"[^}]*\}', '', kramdown)
    # Remove {: updated="..." id="..."} variations
    cleaned = re.sub(r'\{:\s*updated="[^"]+"[^}]*\}', '', cleaned)
    # Remove {{{row and }}} wrappers
    cleaned = re.sub(r'\{\{\{row\n?', '', cleaned)
    cleaned = re.sub(r'\}\}\}', '', cleaned)
    # Remove list item IDs like {: id="..."}
    cleaned = re.sub(r'\s*\{:\s*[^}]+\}', '', cleaned)
    # Clean up excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def fetch_siyuan_notes(
    block_id: Annotated[str, "The unique 22-character Siyuan block ID to fetch."],
) -> str:
    """Fetch markdown content from a Siyuan Notes document or block.

    This tool retrieves knowledge from the local Siyuan Notes instance,
    keeping all data local (no cloud APIs, privacy-preserving).
    """
    headers = (
        {"Authorization": f"Token {config.SIYUAN_API_TOKEN}"}
        if config.SIYUAN_API_TOKEN
        else {}
    )
    payload = {"id": block_id}

    try:
        response = requests.post(
            config.SIYUAN_API_URL, headers=headers, json=payload, timeout=10
        )
        response_data = response.json()

        if response_data.get("code") == 0:
            data = response_data.get("data", {})
            # Clean the kramdown content for better readability
            if "kramdown" in data:
                data["kramdown"] = _clean_kramdown(data["kramdown"])
            return json.dumps(data, indent=2, ensure_ascii=False)
        return f"Siyuan Error: {response_data.get('msg')}"
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Siyuan. Is it running?"
    except Exception as e:
        return f"Error fetching notes: {str(e)}"


def push_to_anki(
    front_text: Annotated[str, "The text for the front of the flashcard."],
    back_text: Annotated[str, "The text for the back of the flashcard."],
) -> str:
    """Push a single flashcard to Anki via the AnkiConnect API."""
    payload = {
        "action": "addNote",
        "version": 6,
        "params": {
            "note": {
                "deckName": config.ANKI_DECK_NAME,
                "modelName": "Basic",
                "fields": {"Front": front_text, "Back": back_text},
                "options": {"allowDuplicate": False},
            }
        },
    }

    try:
        response = requests.post(config.ANKI_CONNECT_URL, json=payload, timeout=10)
        response_data = response.json()

        if response.status_code == 200 and response_data.get("error") is None:
            return f"Card added: {response_data['result']}"
        return f"Failed: {response_data.get('error', 'Unknown')}"
    except requests.exceptions.ConnectionError:
        return "Error: Anki not running"
    except Exception as e:
        return f"Error: {str(e)}"


def push_cards_batch(
    cards_json: Annotated[
        str,
        'JSON string: {"cards": [{"front": "...", "back": "..."}]}'
    ],
) -> str:
    """Push multiple flashcards to Anki in one batch.
    
    Use this to save all approved cards at once.
    """
    import json as json_module
    
    try:
        data = json_module.loads(cards_json)
        cards = data.get('cards', [])
    except json_module.JSONDecodeError:
        return "Error: Invalid JSON"
    
    if not cards:
        return "Error: No cards found in JSON"
    
    results = []
    for i, card in enumerate(cards, 1):
        front = card.get('front', card.get('question', ''))
        back = card.get('back', card.get('answer', ''))
        if front and back:
            result = push_to_anki(front, back)
            results.append(f"Card {i}: {result}")
    
    return f"Saved {len(results)} cards:\n" + "\n".join(results)
