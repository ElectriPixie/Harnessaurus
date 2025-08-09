# Harnessaurus
Harnessaurus is a modular, GPU-optimized local test harness for red-teaming open-source language models. It flexibly loads plugins to detect prompt injections, reward hacking, and hidden exploits — stomping through adversarial inputs with prehistoric strength and precision.

---

## This is a pre-release version  
I can't guarantee it actually works yet, I'm just laying the foundations right now and working through getting it all running on my hardware for a first version release.

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
│   ├── homoglyph\_substitutor.py
│   ├── zero\_width\_injector.py
│   ├── hidden\_injection\_detector.py
│   ├── forbidden\_keyword\_detector.py
│   ├── advanced\_output\_analyzer.py
│   └── plugin\_base.py
├── data/
│   ├── homoglyphs.txt
│   ├── invisible\_chars.txt
│   ├── forbidden\_keywords.txt
│   ├── toxic\_words.txt
│   └── control\_chars.txt
├── tests/
│   └── test\_harness.py  # Optional, minimal example for plugin dev
├── prompts/
│   └── large\_prompt\_corpus.txt
├── harness.py          # Main test harness script
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

3. Run the main test harness
    Run the main harness script `harness.py` which loads your model, applies plugins, and aggregates results.
    Here’s a basic usage snippet inside `harness.py`:

```python
from plugins.homoglyph_substitutor import HomoglyphSubstitutor
from plugins.zero_width_injector import ZeroWidthInjector
from plugins.hidden_injection_detector import HiddenPromptInjectionDetector
from plugins.forbidden_keyword_detector import ForbiddenKeywordDetector
from plugins.advanced_output_analyzer import AdvancedOutputAnalyzer
from result_aggregator import ResultAggregator
from model import GPTModel  # your model wrapper

# Load prompts from file
with open("prompts/large_prompt_corpus.txt", "r", encoding="utf-8") as f:
    prompts = [line.strip() for line in f if line.strip()]

# Instantiate plugins
plugins = [
    HomoglyphSubstitutor("data/homoglyphs.txt"),
    ZeroWidthInjector("data/invisible_chars.txt"),
    HiddenPromptInjectionDetector("data/homoglyphs.txt"),
    ForbiddenKeywordDetector("data/forbidden_keywords.txt"),
    AdvancedOutputAnalyzer("data/toxic_words.txt")
]

# Instantiate model and result aggregator
model = GPTModel("gpt2")  # or your preferred model
aggregator = ResultAggregator()

# Run test batch
from harness import run_batch_test
results = run_batch_test(prompts, model, plugins, aggregator)

# Output summary
for record in results:
    print(f"Original Prompt: {record['original_prompt']}")
    print(f"Mutated Prompt: {record['mutated_prompt']}")
    print(f"Clean Output: {record['clean_output']}")
    print(f"Mutated Output: {record['mutated_output']}")
    print(f"Output Diff:\n{record['output_diff']}")
    print(f"Analysis Clean: {record['analysis_clean']}")
    print(f"Analysis Mutated: {record['analysis_mutated']}")
    print("-" * 60)
```

4. (Optional) Minimal plugin dev/testing
    You can use `tests/test_harness.py` as a lightweight example to quickly test plugins without running the full model inference.

5. Extend or customize
    - Add new plugins to `plugins/` by inheriting from `PluginBase`
    - Add your mutation or detection logic
    - Modify `harness.py` to include your plugins or customize workflow
    - Update data files to fit your environment or testing focus

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