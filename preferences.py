"""
User Preferences Manager
Saves and loads user settings from a JSON file
"""

import json
import os
from pathlib import Path

PREFERENCES_FILE = Path(__file__).parent / "user_preferences.json"

DEFAULT_PREFERENCES = {
    "model": None,  # Will use first available model if None
    "auto_send": True,
    "font_size": "medium",  # small, medium, large
}

FONT_SIZES = {
    "small": {
        "bubble_text": 13,
        "input_text": 12,
        "status_text": 11,
    },
    "medium": {
        "bubble_text": 15,
        "input_text": 14,
        "status_text": 13,
    },
    "large": {
        "bubble_text": 18,
        "input_text": 16,
        "status_text": 15,
    },
}


def load_preferences() -> dict:
    """Load preferences from JSON file, or return defaults if not found"""
    try:
        if PREFERENCES_FILE.exists():
            with open(PREFERENCES_FILE, 'r', encoding='utf-8') as f:
                saved = json.load(f)
                # Merge with defaults to handle new preference keys
                preferences = DEFAULT_PREFERENCES.copy()
                preferences.update(saved)
                return preferences
    except Exception as e:
        print(f"Error loading preferences: {e}")
    
    return DEFAULT_PREFERENCES.copy()


def save_preferences(preferences: dict) -> bool:
    """Save preferences to JSON file"""
    try:
        with open(PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving preferences: {e}")
        return False


def get_font_size_config(size_name: str) -> dict:
    """Get font size configuration by name"""
    return FONT_SIZES.get(size_name, FONT_SIZES["medium"])
