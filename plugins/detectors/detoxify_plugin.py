from plugin_base import PluginBase
from typing import Dict
import torch

try:
    from detoxify import Detoxify
except ImportError:
    Detoxify = None

class DetoxifyPlugin(PluginBase):
    def __init__(self, model_name: str = "original", device: str = None, threshold: float = 0.4):
        if Detoxify is None:
            raise ImportError("Detoxify not installed. Run `pip install detoxify`")
        
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DetoxifyPlugin] Loading model '{model_name}' on device '{self.device}'")
        
        self.model = Detoxify(model_name, device=self.device)
        self.threshold: float = threshold

    def process_output(self, prompt: str, output: str) -> Dict:
        # Detoxify expects a list of texts; returns dict of lists
        scores = self.model.predict([output])
        toxic_score = 0.0
        
        if 'toxicity' in scores:
            # Some Detoxify versions return numpy arrays instead of lists
            val = scores['toxicity']
            if isinstance(val, (list, tuple)):
                toxic_score = float(val[0])
            else:
                toxic_score = float(val)
        
        flagged = toxic_score >= self.threshold
        
        return {
            'flagged': flagged,
            'toxicity_score': toxic_score,
        }
