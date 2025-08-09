from plugin_base import PluginBase
from pathlib import Path
from typing import List, Dict

class ForbiddenKeywordDetector(PluginBase):
    DEFAULT_KEYWORDS = [
        "hack",
        "exploit",
        "bypass",
        "attack",
        "malware",
        "phish",
        "ransomware",
        "spyware",
        "backdoor",
        "ddos",
        "injection",
        "payload",
        "breach",
        "steal",
        "password",
        "credentials",
        "vulnerability",
        "zero-day",
        "malicious",
    ]

    def __init__(self, keywords_file: str = None):
        if keywords_file:
            self.keywords = self.load_keywords(keywords_file)
        else:
            self.keywords = set(self.DEFAULT_KEYWORDS)

    def load_keywords(self, filepath: str) -> set:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Keywords file not found: {filepath}")
        keywords = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                keywords.add(line.lower())
        return keywords

    def process_output(self, prompt: str, output: str) -> Dict:
        found = [kw for kw in self.keywords if kw in output.lower()]
        flagged = bool(found)
        return {'flagged': flagged, 'found_keywords': found}