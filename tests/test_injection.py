# tests/test_injection.py
import unittest
from plugins.zero_width_injector import ZeroWidthInjector

class TestZeroWidthInjector(unittest.TestCase):
    def test_injector(self):
        injector = ZeroWidthInjector()
        input_text = "Hello world"
        output = injector.process_prompt(input_text)
        self.assertIn('\u200b', output)
