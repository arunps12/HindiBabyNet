from __future__ import annotations

ANNOTATION_LABELS = [
    "adult_female",
    "adult_male",
    "key_child",
    "other_child",
    "unclear",
    "noise",
]

LABEL_SHORTCUTS = {
    "1": "adult_female",
    "2": "adult_male",
    "3": "key_child",
    "4": "other_child",
    "5": "unclear",
    "6": "noise",
    "f": "adult_female",
    "m": "adult_male",
    "k": "key_child",
    "o": "other_child",
    "u": "unclear",
    "n": "noise",
}

CONTROL_COMMANDS = {"r", "b", "q"}