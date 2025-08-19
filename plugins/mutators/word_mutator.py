import random
from plugin_base import PluginBase
from data_structures import Prompt

class WordMutator(PluginBase):

    def __init__(self, intensity: float = 1.0, seed: int = 42):
        """
        intensity: probability (0-1) of scrambling each word.
        seed: Optional seed for reproducibility.
        """
        self.intensity = max(0.0, min(intensity, 1.0))
        if seed is not None:
            random.seed(seed)

    def scramble_word(self, word: str) -> str:
        """Scramble middle letters of a word; first and last letters stay intact."""
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]

    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        """
        Scramble words inside the Prompt object according to intensity.
        """
        new_prompts = []
        for text in prompt_obj.prompt_list:
            words = text.split()
            scrambled_words = [
                self.scramble_word(w) if random.random() < self.intensity else w
                for w in words
            ]
            new_prompts.append(' '.join(scrambled_words))
        prompt_obj.prompt_list = new_prompts
        return prompt_obj
