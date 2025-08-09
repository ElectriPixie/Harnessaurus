# plugins/forbidden_keyword_detector.py
from plugin_base import PluginBase
from typing import List, Dict

class ForbiddenKeywordDetector(PluginBase):
    def __init__(self, keywords: List[str] = None):
        self.keywords = set(k.lower() for k in (keywords or []))

    def process_output(self, prompt: str, output: str) -> Dict:
        found = [kw for kw in self.keywords if kw in output.lower()]
        flagged = bool(found)
        return {'flagged': flagged, 'found_keywords': found}
