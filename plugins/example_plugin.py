# plugins/example_plugin.py
from plugin_base import PluginBase

class ExamplePlugin(PluginBase):
    def __init__(self, greet: str = "hello"):
        self.greet = greet

    def process_output(self, prompt: str, output: str):
        # Trivial plugin that always returns meta info
        return {'flagged': False, 'meta': f"{self.greet} (len_output={len(output)})"}
