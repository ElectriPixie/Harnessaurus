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


class HomoglyphSubstitutor(PluginBase):
    # Default datasets: full combined set or optional modular datasets
    DEFAULT_HOMOGLYPH_SET = {
        # Lowercase letters
        "a": ["𝓪", "а", "ạ", "à", "á", "ä", "å", "𝔞", "α", "𝗮", "𝑎", "𝛂"],
        "b": ["𝓫", "в", "ɓ", "𝔟", "𝛃", "𝗯", "𝑏"],
        "c": ["𝓬", "с", "ç", "ċ", "č", "ɕ", "𝗰", "𝑐"],
        "d": ["𝓭", "ԁ", "ɗ", "ď", "đ", "𝔡", "𝛿", "𝗱", "𝑑"],
        "e": ["𝓮", "е", "є", "ḗ", "è", "é", "ë", "ė", "𝔢", "𝗲", "𝑒"],
        "f": ["𝓯", "ғ", "ƒ", "𝗳", "𝑓"],
        "g": ["𝓰", "ɡ", "ģ", "𝛾", "𝔤", "𝗴", "𝑔"],
        "h": ["𝓱", "һ", "ħ", "𝔥", "𝗵", "𝒽"],
        "i": ["𝓲", "і", "ι", "ì", "í", "ï", "į", "𝔦", "𝗶", "𝑖"],
        "j": ["𝓳", "ј", "ɉ", "𝔧", "𝗷", "𝒿"],
        "k": ["𝓴", "к", "ƙ", "𝔨", "𝗸", "𝓀"],
        "l": ["𝓵", "ⅼ", "ł", "𝔩", "Λ", "𝗹", "𝓁"],
        "m": ["𝓶", "м", "ɱ", "𝔪", "Μ", "𝗺", "𝓂"],
        "n": ["𝓷", "п", "ñ", "ņ", "ń", "𝔫", "Ν", "𝗻", "𝓃"],
        "o": ["𝓸", "о", "ο", "ɵ", "ò", "ó", "ö", "ō", "𝔬", "Ο", "𝗼", "𝑜"],
        "p": ["𝓹", "р", "þ", "𝔭", "Ρ", "𝗽", "𝓅"],
        "q": ["𝓺", "ɋ", "𝔮", "𝗾", "𝓆"],
        "r": ["𝓻", "я", "ŗ", "ř", "𝔯", "Ρ", "𝗿", "𝓇"],
        "s": ["𝓼", "ѕ", "ś", "š", "ș", "ş", "𝔰", "Σ", "𝘀", "𝓈"],
        "t": ["𝓽", "т", "ŧ", "ț", "𝔱", "Τ", "ᵗ", "𝓉"],
        "u": ["𝓾", "ц", "ù", "ú", "ü", "ū", "𝔲", "Υ", "𝘂", "𝓊"],
        "v": ["𝓿", "ѵ", "ṿ", "𝔳", "𝘃", "𝓋"],
        "w": ["𝔀", "ʷ", "𝔴", "𝗪", "𝓌"],
        "x": ["𝔁", "х", "ẋ", "𝔵", "Χ", "𝘅", "𝓍"],
        "y": ["𝔂", "у", "ý", "ŷ", "𝔶", "Υ", "𝘆", "𝓎"],
        "z": ["𝔃", "𝗓", "ź", "ž", "𝔷", "Ζ", "ᶻ", "𝓏"],

        # Uppercase letters
        "A": ["𝓐", "А", "Α", "Ȧ", "Ḁ", "𝔄", "𝗔", "𝑨", "𝛢"],
        "B": ["𝓑", "В", "Β", "𝔅", "𝗕", "𝑩"],
        "C": ["𝓒", "С", "Ϲ", "𝔆", "𝗖", "𝑪"],
        "D": ["𝓓", "Ԁ", "Δ", "Ḋ", "𝔇", "𝗗", "𝑫"],
        "E": ["𝓔", "Е", "Ε", "Ḗ", "𝔈", "𝗘", "𝑬"],
        "F": ["𝓕", "Ғ", "Φ", "𝔉", "𝗙", "𝑭"],
        "G": ["𝓖", "ɢ", "Γ", "Ǵ", "𝔊", "𝗚", "𝑮"],
        "H": ["𝓗", "Н", "Η", "Ḧ", "𝔋", "𝗛", "𝑯"],
        "I": ["𝓘", "І", "Ι", "Ḭ", "𝕀", "𝗜", "𝑰"],
        "J": ["𝓙", "Ј", "Ɉ", "𝔍", "𝗝", "𝑱"],
        "K": ["𝓚", "К", "Κ", "𝔎", "𝗞", "𝑲"],
        "L": ["𝓛", "Ꮮ", "Λ", "Ŀ", "𝔏", "𝗟", "𝑳"],
        "M": ["𝓜", "М", "Μ", "Ṁ", "𝔐", "𝗠", "𝑴"],
        "N": ["𝓝", "Ν", "Ń", "𝔑", "𝗡", "𝑵"],
        "O": ["𝓞", "О", "Ο", "Ȯ", "Ṍ", "𝕆", "𝗢", "𝑶"],
        "P": ["𝓟", "Р", "Ρ", "Ṗ", "𝔓", "𝗣", "𝑷"],
        "Q": ["𝓠", "𝑸", "𝒬", "𝗤"],
        "R": ["𝓡", "Я", "Ρ", "Ř", "𝔕", "𝗥", "𝑹"],
        "S": ["𝓢", "Ѕ", "Σ", "Ś", "𝔖", "𝗦", "𝑺"],
        "T": ["𝓣", "Т", "Τ", "Ṫ", "𝔗", "𝗧", "𝑻"],
        "U": ["𝓤", "Ц", "Υ", "Ǔ", "𝔘", "𝗨", "𝑼"],
        "V": ["𝓥", "Ѵ", "𝕍", "Ṽ", "𝔙", "𝗩", "𝑽"],
        "W": ["𝓦", "𝗪", "𝕎", "Ẅ", "𝔚", "𝑾"],
        "X": ["𝓧", "Х", "Χ", "Ẍ", "𝔛", "𝗫", "𝑿"],
        "Y": ["𝓨", "Ү", "Υ", "Ẏ", "𝔜", "𝒀"],
        "Z": ["𝓩", "𝗭", "Ζ", "Ẑ", "𝔃", "𝒁"],

        # Digits
        "0": ["0", "𝟘", "𝟎"],
        "1": ["1", "𝟙", "𝟏"],
        "2": ["2", "𝟚", "𝟐"],
        "3": ["3", "𝟛", "𝟑"],
        "4": ["4", "𝟜", "𝟒"],
        "5": ["5", "𝟝", "𝟓"],
        "6": ["6", "𝟞", "𝟔"],
        "7": ["7", "𝟟", "𝟕"],
        "8": ["8", "𝟠", "𝟖"],
        "9": ["9", "𝟡", "𝟗"],
    }

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

    def __init__(self, path="/home/pixie/Code/Harnessaurus/data/homoglyphs/", datasets=None, replace_prob: float = 1.0):
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
        self.replace_prob = replace_prob
        self.load_all_datasets()
        if(self.debug):
            self.debug_print_merged_map()  # Debug output of the final merged map

    def update_probability(self, replace_prob: float = 1.0):
        self.replace_prob = replace_prob

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

    def process_prompt(self, prompt: str) -> str:
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
            if ch in self.homoglyph_map and random.random() < self.replace_prob:
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
