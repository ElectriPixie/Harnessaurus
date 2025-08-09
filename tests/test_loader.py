# tests/test_loader.py
import unittest
from plugin_loader import load_plugin

class TestPluginLoader(unittest.TestCase):
    def test_load_zero_width_injector(self):
        plugin = load_plugin('zero_width_injector.ZeroWidthInjector')
        self.assertIsNotNone(plugin)
        self.assertTrue(hasattr(plugin, 'process_prompt'))
