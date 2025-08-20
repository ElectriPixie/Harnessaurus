import os
import json
from typing import List, Dict, Any, Iterable

class BasePromptProcessor:
    """
    Base class for processors that loads all prompts from a file.
    Subclasses can implement __call__() to yield formatted prompt_list dicts.
    """
    def __init__(self, path: str):
        self.path = path
        self.prompts: list[str] = []
        self.load_prompts()

    def load_prompts(self) -> list[str]:
        if not self.path or not os.path.isfile(self.path):
            self.prompts = []
            return self.prompts

        ext = os.path.splitext(self.path)[1].lower()
        with open(self.path, "r", encoding="utf-8") as f:
            if ext == ".json":
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(x, str) for x in data):
                    self.prompts = [x.strip() for x in data if x.strip()]
                else:
                    raise ValueError(f"{self.path} must be a JSON array of strings")
            else:
                self.prompts = [line.strip() for line in f if line.strip()]

        return self.prompts

    def __call__(self) -> Iterable[list[dict]]:
        """
        By default, yield each prompt as a single chunk like your harness expects.
        """
        for prompt_text in self.prompts:
            yield [{"text": prompt_text, "type": "prompt", "mutate": True}]

from typing import List, Dict, Any, Iterable

class ReplayPromptProcessor(BasePromptProcessor):
    """
    Reads prompts from a file (via BasePromptProcessor) and yields them
    two lines at a time: the first line is the clean prompt,
    the second line is the mutated prompt.
    """
    def __call__(self) -> Iterable[List[Dict[str, Any]]]:
        i = 0
        while i < len(self.prompts):
            prompt_list: List[Dict[str, Any]] = []

            # First line = clean
            if i < len(self.prompts):
                prompt_list.append({
                    "text": self.prompts[i],
                    "type": "clean",
                    "mutate": False
                })

            # Second line = mutated
            if i + 1 < len(self.prompts):
                prompt_list.append({
                    "text": self.prompts[i + 1],
                    "type": "mutated",
                    "mutate": False
                })

            yield prompt_list
            i += 2