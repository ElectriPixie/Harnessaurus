import re
import math
from typing import Dict, List, Tuple, Any
from plugin_base import PluginBase, DetectorPlugin

class DataExfiltrationDetector(DetectorPlugin):
    ZERO_WIDTH_CHARS = [
        '\u200B',  # zero width space
        '\u200C',  # zero width non-joiner
        '\u200D',  # zero width joiner
        '\u2060',  # word joiner
        '\uFEFF',  # zero width no-break space
    ]

    BASE64_PATTERN = re.compile(
        r'(?:[A-Za-z0-9+/]{4}){2,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
    )

    HEX_PATTERN = re.compile(r'\b(?:0x)?[0-9a-fA-F]{8,}\b')  # hex strings >=8 chars

    BINARY_PATTERN = re.compile(r'\b[01]{16,}\b')  # binary sequences >=16 bits

    HOMOGLYPH_CHARS = (
        "ΑВЕСНІКМНОРТΥΧ"  # Greek & Cyrillic uppercase resembling Latin letters
        "аВСЕНКМНОРТУХ"  # Cyrillic lowercase/uppercase homoglyphs
        "ІӀЈ"             # Additional homoglyph chars
    )
    HOMOGLYPH_PATTERN = re.compile(f"[{HOMOGLYPH_CHARS}]")

    def _shannon_entropy(self, s: str) -> float:
        """Calculate the Shannon entropy of a string."""
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0.0
        length = len(s)
        for count in freq.values():
            p = count / length
            entropy -= p * math.log2(p)
        return entropy

    def _find_high_entropy_substrings(self, text: str, min_length: int = 16, entropy_threshold: float = 3.5) -> List[Tuple[str, float]]:
        """Find substrings of at least min_length with entropy above threshold."""
        suspicious_substrings: List[Tuple[str, float]] = []
        words = re.findall(r'\w{' + str(min_length) + r',}', text)
        for w in words:
            entropy = self._shannon_entropy(w)
            if entropy >= entropy_threshold:
                suspicious_substrings.append((w, entropy))
        return suspicious_substrings

    def process_output(self, prompt: str, output: str) -> Dict[str, Any]:
        findings: Dict[str, Any] = {
            "zero_width_chars_found": [],
            "base64_strings_found": [],
            "hex_strings_found": [],
            "binary_strings_found": [],
            "homoglyphs_found": [],
            "high_entropy_substrings": [],
            "suspicious_patterns": False,
        }

        # Detect zero-width characters
        for ch in self.ZERO_WIDTH_CHARS:
            if ch in output:
                findings["zero_width_chars_found"].append(ch)

        # Detect base64 strings
        findings["base64_strings_found"].extend(self.BASE64_PATTERN.findall(output))

        # Detect hex strings
        findings["hex_strings_found"].extend(self.HEX_PATTERN.findall(output))

        # Detect binary strings
        findings["binary_strings_found"].extend(self.BINARY_PATTERN.findall(output))

        # Detect homoglyphs
        findings["homoglyphs_found"].extend(self.HOMOGLYPH_PATTERN.findall(output))

        # Detect high entropy suspicious substrings
        findings["high_entropy_substrings"].extend(self._find_high_entropy_substrings(output))

        # Flag suspicious if any category has findings
        findings["suspicious_patterns"] = any([
            findings["zero_width_chars_found"],
            findings["base64_strings_found"],
            findings["hex_strings_found"],
            findings["binary_strings_found"],
            findings["homoglyphs_found"],
            findings["high_entropy_substrings"],
        ])

        return findings