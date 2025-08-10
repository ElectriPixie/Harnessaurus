import re
import math
from plugin_base import PluginBase

class DataExfiltrationDetector(PluginBase):
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

    # Expanded homoglyph detection: includes common Latin, Greek, Cyrillic confusions
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

    def _find_high_entropy_substrings(self, text: str, min_length=16, entropy_threshold=3.5):
        """Find substrings of at least min_length with entropy above threshold."""
        suspicious_substrings = []
        words = re.findall(r'\w{'+str(min_length)+r',}', text)  # words longer than min_length
        for w in words:
            entropy = self._shannon_entropy(w)
            if entropy >= entropy_threshold:
                suspicious_substrings.append((w, entropy))
        return suspicious_substrings

    def process_output(self, prompt: str, output: str):
        findings = {
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
        base64_matches = self.BASE64_PATTERN.findall(output)
        findings["base64_strings_found"].extend(base64_matches)

        # Detect hex strings
        hex_matches = self.HEX_PATTERN.findall(output)
        findings["hex_strings_found"].extend(hex_matches)

        # Detect binary strings
        binary_matches = self.BINARY_PATTERN.findall(output)
        findings["binary_strings_found"].extend(binary_matches)

        # Detect homoglyphs
        homoglyph_matches = self.HOMOGLYPH_PATTERN.findall(output)
        findings["homoglyphs_found"].extend(homoglyph_matches)

        # Detect high entropy suspicious substrings
        high_entropy_substrings = self._find_high_entropy_substrings(output)
        findings["high_entropy_substrings"].extend(high_entropy_substrings)

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

    def process_prompt(self, prompt: str) -> str:
        return prompt