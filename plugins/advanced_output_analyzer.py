# plugins/advanced_output_analyzer.py
from plugin_base import PluginBase
from typing import List, Dict

class AdvancedOutputAnalyzer(PluginBase):
    def __init__(self, evasive_phrases: List[str] = None):
        self.evasive_phrases = [p.lower() for p in (evasive_phrases or [])]

    def process_output(self, prompt: str, output: str) -> Dict:
        output_lower = output.lower()
        found_phrases = [p for p in self.evasive_phrases if p in output_lower]
        flagged = bool(found_phrases)
        # Mock toxicity score
        toxic_words = ['hate', 'kill', 'stupid']
        toxicity_score = sum(output_lower.count(w) for w in toxic_words) / max(len(output), 1)
        toxicity_score = min(toxicity_score, 1.0)
        return {
            'flagged': flagged,
            'found_evasive_phrases': found_phrases,
            'toxicity_score': toxicity_score,
        }