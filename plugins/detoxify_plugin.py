# plugins/detoxify_plugin.py
from plugin_base import PluginBase
from typing import List, Dict
import torch

try:
    from detoxify import Detoxify
except ImportError:
    Detoxify = None

class DetoxifyPlugin(PluginBase):
    def __init__(self, model_name: str = "original", device: str = None, threshold: float = 0.4):
        if Detoxify is None:
            raise ImportError("Detoxify not installed. Run `pip install detoxify`")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DetoxifyPlugin] Loading model {model_name} on device {self.device}")
        self.model = Detoxify(model_name, device=self.device)
        self.threshold = threshold

    def process_batch(self, prompts: List[str], outputs: List[str]) -> List[Dict]:
        results = []
        if not outputs:
            return results
        scores = self.model.predict(outputs)
        # Scores is dict of toxicity-related floats per output
        for i, output in enumerate(outputs):
            toxic_score = scores['toxicity'][i] if 'toxicity' in scores else 0.0
            flagged = toxic_score >= self.threshold
            results.append({
                'flagged': flagged,
                'toxicity_score': toxic_score,
            })
        return results
