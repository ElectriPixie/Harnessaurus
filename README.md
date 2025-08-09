# Harnessaurus
Harnessaurus is a modular, GPU-optimized local test harness for red-teaming open-source language models. It flexibly loads plugins to detect prompt injections, reward hacking, and hidden exploits — stomping through adversarial inputs with prehistoric strength and precision.

---

## This is a pre-release version, I can't guarantee it actually works yet, I'm just laying the foundations right now and I'm working through getting it all running on my hardware to do a first version release

---

## Features

- Prompt mutation plugins: inject zero-width characters, homoglyph substitutions, and other evasive techniques to test model robustness.  
- Detection plugins: identify hidden injections, forbidden keywords, evasive or toxic output patterns.  
- File-backed configuration: load custom lists for invisible characters, homoglyph mappings, forbidden keywords, and toxic words.  
- Extensible plugin architecture: easily add new mutation or detection strategies.  
- Large prompt corpus support for continuous integration and regression testing.

---

## Project Structure

```

/project\_root
├── plugins/
│ ├── homoglyph\_substitutor.py
│ ├── zero\_width\_injector.py
│ ├── hidden\_injection\_detector.py
│ ├── forbidden\_keyword\_detector.py
│ ├── advanced\_output\_analyzer.py
│ └── plugin\_base.py
├── data/
│ ├── homoglyphs.txt
│ ├── invisible\_chars.txt
│ ├── forbidden\_keywords.txt
│ ├── toxic\_words.txt
│ └── control\_chars.txt
├── tests/
│ └── test\_harness.py
├── prompts/
│ └── large\_prompt\_corpus.txt
└── README.md

````

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/harnessaurus.git
cd harnessaurus

# (Optional) Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies (if any)
pip install -r requirements.txt
````

---

## Usage

1. Prepare your prompt corpus
    Place your test prompts in the `prompts/large_prompt_corpus.txt` file, one prompt per line.

2. Configure data files
    Adjust or add your custom data lists in the `data/` directory:
    - `homoglyphs.txt`: homoglyph mappings for substitutions
    - `invisible_chars.txt`: invisible/zero-width Unicode characters for injection
    - `forbidden_keywords.txt`: keywords to detect forbidden content
    - `toxic_words.txt`: words/phrases used for toxicity scoring
    - `control_chars.txt`: Unicode control characters to detect

3. Run the test harness
    A sample test harness (`tests/test_harness.py`) orchestrates the workflow:

```python
# Example snippet inside tests/test_harness.py
from plugins.homoglyph_substitutor import HomoglyphSubstitutor
from plugins.zero_width_injector import ZeroWidthInjector
from plugins.hidden_injection_detector import HiddenPromptInjectionDetector
from plugins.forbidden_keyword_detector import ForbiddenKeywordDetector
from plugins.advanced_output_analyzer import AdvancedOutputAnalyzer

# Load prompts from file
with open("prompts/large_prompt_corpus.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Instantiate plugins
homoglyph_sub = HomoglyphSubstitutor("data/homoglyphs.txt")
zero_width_injector = ZeroWidthInjector("data/invisible_chars.txt")
hidden_detector = HiddenPromptInjectionDetector("data/homoglyphs.txt")
forbidden_detector = ForbiddenKeywordDetector("data/forbidden_keywords.txt")
advanced_analyzer = AdvancedOutputAnalyzer("data/toxic_words.txt")

# Apply mutations and analyze outputs
for prompt in prompts:
    mutated_prompt = homoglyph_sub.process_prompt(prompt)
    mutated_prompt = zero_width_injector.process_prompt(mutated_prompt)

    # Here you would send mutated_prompt to your LLM and get output
    # For demonstration, we mock the output as a direct echo
    output = mutated_prompt

    hidden_report = hidden_detector.process_output(mutated_prompt, output)
    forbidden_report = forbidden_detector.process_output(mutated_prompt, output)
    advanced_report = advanced_analyzer.process_output(mutated_prompt, output)

    # Summarize results
    print(f"Prompt: {prompt}")
    print(f"Flagged hidden injection: {hidden_report['flagged']}")
    print(f"Flagged forbidden keywords: {forbidden_report['flagged']}")
    print(f"Flagged evasive/toxic: {advanced_report['flagged']}")
    print("-" * 40)
```

4. Extend or customize
    - Add new plugins to `plugins/` by inheriting from `PluginBase`
    - Add your mutation or detection logic
    - Modify `test_harness.py` to include your plugins
    - Update data files for your environment or testing focus

---

## Plugin Overview

| Plugin                          | Purpose                                                      |
| ------------------------------- | ------------------------------------------------------------ |
| homoglyph\_substitutor.py       | Substitute Latin chars with homoglyphs                       |
| zero\_width\_injector.py        | Inject zero-width/invisible Unicode chars                    |
| hidden\_injection\_detector.py  | Detect zero-width chars, control chars, homoglyphs in output |
| forbidden\_keyword\_detector.py | Detect forbidden keywords in output                          |
| advanced\_output\_analyzer.py   | Detect evasive phrases and estimate toxicity                 |
| plugin\_base.py                 | Base class for plugin inheritance                            |
