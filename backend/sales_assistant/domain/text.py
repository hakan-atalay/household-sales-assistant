from __future__ import annotations

import re
import unicodedata

_SEPARATORS = re.compile(r"[\s_\-/]+")
_TURKISH_FOLD = str.maketrans({"ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ğ": "g", "Ğ": "g", "ç": "c", "Ç": "c",
                               "ö": "o", "Ö": "o", "ü": "u", "Ü": "u"})


def fold(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", str(text).translate(_TURKISH_FOLD))
    ascii_only = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    return _SEPARATORS.sub(" ", ascii_only.lower()).strip()
