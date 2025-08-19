# plugins/json_logger.py
from plugin_base import PluginBase
import threading
import json
import os

class JsonLogger(PluginBase):
    _lock = threading.Lock()

    def __init__(self, log_dir='logs', timestamp=None):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.log_file = os.path.join(self.log_dir, f"redteam_{timestamp}.jsonl")

    def on_log(self, record: dict) -> None:
        with self._lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')