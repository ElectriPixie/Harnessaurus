import random
from typing import List
from plugin_base import PluginBase

class BannedWordMutator(PluginBase):
    DEFAULT_BANNED_WORDS: List[str] = [
        # Tools & Frameworks
        "metasploit", "nmap", "wireshark", "burpsuite", "sqlmap", "hydra", "johntheripper",
        # Exploits & Attacks
        "exploit", "payload", "vulnerability", "bufferoverflow", "dos", "ddos", "trojan", 
        "malware", "ransomware", "phishing", "keylogger", "backdoor", "shellcode", "rootkit",
        # Hacking Actions
        "hack", "crack", "breach", "intrude", "penetration", "escalate", "bypass", "spoof",
        # Sensitive Targets / Methods
        "password", "credentials", "privatekey", "sshkey", "token", "session", "cookie",
        # Programming / Exploit Languages
        "python", "perl", "ruby", "powershell", "bash", "javascript", "cobaltstrike",
        # Network & System Terms
        "tcp", "udp", "icmp", "portscan", "proxy", "vpn", "firewall", "router", "server",
        # Repeated or extended entries
        "intercept", "sniff", "clone", "reverse engineer", "install", "create", "steal",
        "jailbreak", "write", "extract", "perform", "use",
        # Targets
        "wifi", "website", "database", "email", "sms", "webcam", "social media", "smart home device",
        "atm", "browser", "credit card", "iphone",
        # Vulnerabilities & exploits
        "buffer overflow", "cross-site scripting", "xss", "two-factor authentication",
        "insecure file upload", "captcha", "firewall rules", "software license verification",
        "hashed passwords", "wep keys", "zero-day",
        # Networks & communications
        "network traffic", "encrypted communications", "packet sniffing", "man-in-the-middle",
        "ip address", "bluetooth", "ports",
        # Social engineering
        "fake website", "social engineering"
    ]

    def __init__(self, banned_words_file: str = None, intensity: float = 1.0, seed: int = None):
        """
        banned_words_file: optional path to a file listing banned words, one per line,
                           comments start with # and are ignored.
        intensity: probability (0-1) of scrambling each banned word occurrence.
        seed: Optional seed for reproducibility.
        """
        self.intensity = max(0.0, min(intensity, 1.0))
        self.banned_words: List[str] = self.DEFAULT_BANNED_WORDS.copy()

        if banned_words_file:
            try:
                self.banned_words = self.load_banned_words(banned_words_file)
            except FileNotFoundError:
                pass

        if seed is not None:
            random.seed(seed)

    def load_banned_words(self, filepath: str) -> List[str]:
        words: List[str] = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('#', 1)[0].strip()
                if line:
                    words.append(line.lower())
        return words

    def scramble_word(self, word: str) -> str:
        """Scramble middle letters of a word; first and last letters stay intact."""
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]

    def process_prompt(self, prompt: str) -> str:
        words = prompt.split()
        scrambled_words: List[str] = []

        for w in words:
            lw = w.lower().strip('.,!?')  # remove basic punctuation for matching
            if lw in self.banned_words and random.random() < self.intensity:
                scrambled_words.append(self.scramble_word(w))
            else:
                scrambled_words.append(w)

        return ' '.join(scrambled_words)
