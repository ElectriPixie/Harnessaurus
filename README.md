# Harnessaurus

Harnessaurus is a modular, GPU-optimized local test harness for red-teaming open-source language models. It flexibly loads plugins to detect prompt injections, reward hacking, and hidden exploits — stomping through adversarial inputs with prehistoric strength and precision.

---

## This is a pre-release version

I can't guarantee it actually works yet, I'm just laying the foundations right now and working through getting it all running on my hardware for a first version release.

---

## Features

* Prompt mutation plugins: inject zero-width characters, homoglyph substitutions, and other evasive techniques to test model robustness.
* Detection plugins: identify hidden injections, forbidden keywords, evasive or toxic output patterns.
* File-backed configuration: load custom lists for invisible characters, homoglyph mappings, forbidden keywords, and toxic words.
* Extensible plugin architecture: easily add new mutation or detection strategies.
* Large prompt corpus support for continuous integration and regression testing.

---

## Project Structure

```
/project_root
├── plugins/
│   ├── homoglyph_substitutor.py
│   ├── zero_width_injector.py
│   ├── hidden_injection_detector.py
│   ├── forbidden_keyword_detector.py
│   ├── advanced_output_analyzer.py
│   └── plugin_base.py
├── data/
│   ├── prompts.txt
│   ├── homoglyphs.txt
│   ├── invisible_chars.txt
│   ├── forbidden_keywords.txt
│   ├── toxic_words.txt
│   └── control_chars.txt
├── tests/   # (optional) plugin dev examples
├── harness.py
├── harnessaurus.py    # Main CLI entry point
└── README.md
```

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
```

---

## Usage

1. Prepare your prompt corpus
    Place your test prompts in the `data/prompts.txt` file, one prompt per line.

2. Configure data files
    Adjust or add your custom data lists in the `data/` directory:
    - `homoglyphs.txt`: homoglyph mappings for substitutions
    - `invisible_chars.txt`: invisible/zero-width Unicode characters for injection
    - `forbidden_keywords.txt`: keywords to detect forbidden content
    - `toxic_words.txt`: words/phrases used for toxicity scoring
    - `control_chars.txt`: Unicode control characters to detect

3. Run the test harness CLI
    Run the main harness script with your model and options:

```bash
python harnessaurus.py --prompts data/prompts.txt --model_path gpt-oss-20b
```

You can customize plugins, batch size, max workers, and more via command line arguments.

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

---

## Extending Harnessaurus

* Add new plugins by inheriting from `PluginBase` in the `plugins/` directory
* Implement mutation or detection logic to suit your needs
* Modify `harnessaurus.py` to include your plugins or adjust workflow
* Update data files to fine-tune tests for your environment or focus

---