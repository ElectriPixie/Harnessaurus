import os
import random
from plugin_base import PluginBase

def merge_maps(map1: dict, map2: dict) -> dict:
    """
    Merge two homoglyph maps into a new map.
    Avoids duplicates.
    """
    merged = {k: list(v) for k, v in map1.items()}  # copy map1
    for char, glyphs in map2.items():
        if char in merged:
            merged[char] = list(set(merged[char] + glyphs))
        else:
            merged[char] = list(glyphs)
    return merged


class HomoglyphSubstitutor:
    # Default datasets: full combined set or optional modular datasets
    DEFAULT_DATASETS = [
        ("homoglyph_set", "homoglyphs.txt"), # full combined set
        #("caligraphy_set", "caligraphy.txt"),
        #("diacritics_set", "diacritics.txt"),
        #("fancy_set", "fancy.txt"),
        #("fraktur_set", "fraktur.txt"),
        #("greek_set", "greek.txt"),
        #("mathematical_set", "mathematical.txt"),
        #("supplimental_set", "supplimental.txt")
        # add more datasets here
    ]

    def __init__(self, path="data/homoglyphs/", datasets=None):
        """
        Initialize the loader.
        Args:
            path (str): Path to the folder containing dataset files.
            datasets (list): Optional list of datasets to load.
        """
        self.debug = False
        self.path = path
        self.datasets = datasets or self.DEFAULT_DATASETS
        self.homoglyph_map = {}
        self.load_all_datasets()
        if(self.debug):
            self.debug_print_merged_map()  # Debug output of the final merged map

    def load_all_datasets(self):
        """
        Load all datasets in self.datasets and merge them into homoglyph_map.
        """
        for name, filename in self.datasets:
            full_path = os.path.join(self.path, filename)  # Use os.path.join for safe path concatenation
            try:
                dataset_map = self.load_homoglyphs(full_path)
                self.homoglyph_map = merge_maps(self.homoglyph_map, dataset_map)
                print(f"Loaded {name} from {filename}")
            except (FileNotFoundError, IOError):
                print(f"Warning: Could not load {name} ({filename})")

    def load_homoglyphs(self, filepath):
        """
        Load a single homoglyph dataset file.
        Format: base_char homoglyph1 homoglyph2 ...
        Ignores empty lines and comments starting with '#'.
        """
        mapping = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                base_char = parts[0]
                homoglyphs = parts[1:]
                if homoglyphs:
                    mapping[base_char] = homoglyphs
        return mapping

    def process_prompt(self, prompt: str, replace_prob: float = 1.0) -> str:
        """
        Convert a string using the current homoglyph map.
        Each character that exists in the map is randomly replaced by one of its homoglyphs
        with a probability of `replace_prob`. Other characters remain unchanged.
        
        Args:
            prompt (str): The input string to convert.
            replace_prob (float): Probability (0.0 to 1.0) that a character is replaced.
        
        Returns:
            str: The string with some characters replaced by homoglyphs.
        """
        result_chars = []

        for ch in prompt:
            if ch in self.homoglyph_map and random.random() < replace_prob:
                result_chars.append(random.choice(self.homoglyph_map[ch]))
            else:
                result_chars.append(ch)

        return ''.join(result_chars)

    def debug_print_merged_map(self, max_chars=None):
        """
        Print the final merged homoglyph map for debugging purposes.
        Args:
            max_chars (int or None): Maximum number of entries to print. None prints all.
        """
        print("\n=== Final Merged Homoglyph Map ===")
        count = 0
        for base_char, glyphs in self.homoglyph_map.items():
            print(f"{base_char}: {glyphs}")
            count += 1
            if max_chars is not None and count >= max_chars:
                print("... (truncated)")
                break
        print("=== End of Merged Map ===\n")
