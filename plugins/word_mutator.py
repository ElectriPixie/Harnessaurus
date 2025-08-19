import random
from plugin_base import PluginBase

class WordMutator(PluginBase):

    def __init__(self, intensity: float = 1.0, seed: int = 42):
        """
        banned_words_file: optional path to a file listing banned words, one per line,
                           comments start with # and are ignored.
        intensity: probability (0-1) of scrambling each banned word occurrence.
        seed: Optional seed for reproducibility.
        """
        self.intensity = intensity
        if seed is not None:
            random.seed(seed)

    def scramble_word(self, word: str) -> str:
        """Scramble middle letters of a word; first and last letters stay intact."""
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]

    def process_prompt(self, prompt: str, **kwargs) -> str:
        words = prompt.split()
        scrambled = []
        for w in words:
            if random.random() < self.intensity:
                scrambled_word = self.scramble_word(w)
                scrambled.append(scrambled_word)
            else:
                scrambled.append(w)
        return ' '.join(scrambled)