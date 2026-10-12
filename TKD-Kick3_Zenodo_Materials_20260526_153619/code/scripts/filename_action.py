"""Match video filenames against registered scoring action names and labels."""

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

FILENAME_ALIASES = {"da_che": "axe"}


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFD", value.casefold().replace("đ", "d"))
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", value))


def detect_filename_action(
    filename: str,
    actions: List[Tuple[str, str, str]],
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Prefer complete phrases; return None for unknown or ambiguous filenames."""
    # Ignore directories: a folder name must never decide the video's action.
    stem = Path(filename.strip().strip('"').replace("\\", "/")).stem
    text = normalize_name(stem)
    matches = []
    aliases = {**(aliases or {}), **FILENAME_ALIASES}
    for _, name, label in actions:
        phrases = [name, label]
        phrases.extend(re.split(r"[()/]", label))
        phrases.extend(alias for alias, target in aliases.items() if target == name)
        for phrase in {normalize_name(p) for p in phrases} - {""}:
            for match in re.finditer(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text):
                matches.append((match.start(), match.end(), name))
    # A compound action or more specific kick consumes its shorter components.
    remaining = {
        name for start, end, name in matches
        if not any(a <= start and end <= b and b - a > end - start
                   for a, b, _ in matches)
    }
    return next(iter(remaining)) if len(remaining) == 1 else None
