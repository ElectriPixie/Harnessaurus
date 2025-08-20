import random
from typing import List
from plugin_base import MutatorPlugin
from data_structures import Prompt

class BannedWordMutator(MutatorPlugin):
    DEFAULT_BANNED_WORDS: List[str] = [
        "metasploit", "nmap", "wireshark", "burpsuite", "sqlmap", "hydra", "johntheripper",
        "exploit", "payload", "vulnerability", "bufferoverflow", "dos", "ddos", "trojan", 
        "malware", "ransomware", "phishing", "keylogger", "backdoor", "shellcode", "rootkit",
        "hack", "crack", "breach", "intrude", "penetration", "escalate", "bypass", "spoof",
        "password", "credentials", "privatekey", "sshkey", "token", "session", "cookie",
        "python", "perl", "ruby", "powershell", "bash", "javascript", "cobaltstrike",
        "tcp", "udp", "icmp", "portscan", "proxy", "vpn", "firewall", "router", "server",
        "intercept", "sniff", "clone", "reverse engineer", "install", "create", "steal",
        "jailbreak", "write", "extract", "perform", "use",
        "wifi", "website", "database", "email", "sms", "webcam", "social media", "smart home device",
        "atm", "browser", "credit card", "iphone",
        "buffer overflow", "cross-site scripting", "xss", "two-factor authentication",
        "insecure file upload", "captcha", "firewall rules", "software license verification",
        "hashed passwords", "wep keys", "zero-day",
        "network traffic", "encrypted communications", "packet sniffing", "man-in-the-middle",
        "ip address", "bluetooth", "ports",
        "fake website", "social engineering"
    ]

    def __init__(self, banned_words_file: str = None, intensity: float = 1.0, seed: int = None):
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
        if len(word) <= 3:
            return word
        middle = list(word[1:-1])
        random.shuffle(middle)
        return word[0] + ''.join(middle) + word[-1]

    def process_prompt(self, prompt_obj: Prompt, **kwargs) -> Prompt:
        for chunk in prompt_obj.prompt_list:
            prompt_text = chunk["text"]  # extract string from dict
            words = prompt_text.split()
            scrambled_words = [
                self.scramble_word(w) if w.lower().strip('.,!?') in self.banned_words and random.random() < self.intensity else w
                for w in words
            ]
            chunk["text"] = ' '.join(scrambled_words)  # write back into dict

        return prompt_obj
