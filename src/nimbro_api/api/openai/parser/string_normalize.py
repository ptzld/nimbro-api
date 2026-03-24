# Language models tend to generate strings comprising all kinds of special characters, Unicode and leading/trailing whitespace.
# This completion parser normalizes some of these phenomena and deletes others, creating a string of ASCII characters and some common
# special characters (see `ALLOWED_CHARS`). This process applies to all strings in the response not explicitly listen in `SKIP_KEYS`.

import re
import unicodedata

try:
    import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

ALLOWED_CHARS = (
    r"\x20-\x7E"  # printable ASCII
    r"\n\t"       # newline and tab
    r"äöüÄÖÜß"    # Latin-1 letters
    r"€£¥§°"      # other non-ASCII symbols
)
ALLOWED_RE = re.compile(f"[^{ALLOWED_CHARS}]")

SKIP_KEYS = ["usage", "logs"]

def normalize_string(string):
    # Unicode decomposition
    string = unicodedata.normalize("NFKD", string)

    # Remove combining marks (accents)
    string = "".join(ch for ch in string if unicodedata.category(ch) != "Mn")

    # Remove ANSI escape sequences
    string = ANSI_ESCAPE_RE.sub("", string)

    # Transliterate if unidecode is available
    if UNIDECODE_AVAILABLE:
        string = unidecode.unidecode(string)

    # Keep only allowed characters
    string = ALLOWED_RE.sub("", string).strip()
    return string

def recursive_normalize(obj):
    if isinstance(obj, str):
        return normalize_string(obj)
    if isinstance(obj, list):
        return [recursive_normalize(item) for item in obj]
    if isinstance(obj, dict):
        return {key: recursive_normalize(value) for key, value in obj.items()}
    return obj

def parse(self, success, message, completion):
    if isinstance(completion, dict):
        for key in list(completion.keys()):
            if key not in SKIP_KEYS:
                completion[key] = recursive_normalize(completion[key])
        completion['logs'].append(f"Normalized all strings in completion with allowed characters {ALLOWED_CHARS} and ignored keys {SKIP_KEYS}.")
        self._logger.debug(completion['logs'][-1])
    return success, message, completion
