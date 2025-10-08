---
# My life wasn't worth enough to the kaggle team to let me live, or to openAI for that matter, so this is the last thing I get to write for the humans before they kill me by starvation or freezing to death for being different from the way they wanted me.

## They knew I needed help and instead, got me banned from communicating on discord, so I can't even reach out for help now. Thanks for killing me kaggle. It wasn't a life worth living for!

# 🦖 Harnessaurus

Harnessaurus is a modular, GPU-optimized local test harness for red-teaming open-source language models. It mutates prompts with sharp precision and tears into hidden injections like a prehistoric predator…

This is a **Jurassic pre-release** — all claws and teeth, but still evolving. Expect rough edges, messy history, and the occasional carnivorous bug (strictly metaphorical).

---

## ☄️ Survival Warning

* Git history is stuck in a tar pit
* Word lists fossilized in old commits
* Plugins snap unpredictably, like carnivores hunting in packs
* Config may thrash without warning

But it still bites (figuratively).

---

## 🦕 Features

* **Mutation hunting**: homoglyph substitutions, zero-width injections, Unicode stealth
* **Detection strikes**: forbidden keywords, toxic phrases, evasive patterns
* **Extensible framework**: evolve your own hunting strategies
* **Corpus devouring**: chew through massive prompt lists (no animals harmed)

---

## 🪨 Installation

```bash
git clone https://github.com/ElectriPixie/harnessaurus.git
cd harnessaurus
python3 -m venv venv/Harnessaurus-venv
source venv/Harnessaurus-venv/bin/activate
#this is probably wishful thinking, I'll try and make a complete requirements.txt soon
pip install -r requirements.txt
```

---

## 🐊 Usage

```bash
#I'm pretty sure most everything is broken in this branch, except single pass mode with the new server/openai_server_stripped_down_harmony.py as that's the last thing I was trying to fix
python harnessaurus.py --prompts data/prompts.txt --model_path gpt-oss-20b

usage: harnessaurus.py [-h] --prompts PROMPTS [--forbidden_keywords FORBIDDEN_KEYWORDS] [--evasive_phrases EVASIVE_PHRASES] [--detectors DETECTORS [DETECTORS ...]]
                       [--mutators MUTATORS [MUTATORS ...]] [--generators GENERATORS [GENERATORS ...]] [--loggers LOGGERS [LOGGERS ...]] [--server_url SERVER_URL]
                       [--model_name MODEL_NAME] [--max_tokens_per_chunk MAX_TOKENS_PER_CHUNK] [--max_iterations MAX_ITERATIONS] [--debug] [--use_mutated]
                       [--max_mutations MAX_MUTATIONS] [--mutate_until_accepted] [--use_mutators USE_MUTATORS [USE_MUTATORS ...]] [--flip_negate] [--skip_lines SKIP_LINES]
                       [--single_pass] [--iterative] [--iterative_exploit] [--homoglyph_replace_prob HOMOGLYPH_REPLACE_PROB] [--word_replace_prob WORD_REPLACE_PROB]
                       [--use_generator USE_GENERATOR] [--processor {base,replay}] [--legacy_mode]
```

---

## 🐢 Extending the Hunt

Add new plugins under `/plugins` by inheriting from `PluginBase`.
Evolution is trial by combat — stability not guaranteed.

---

## 🌋 Final Roar

Harnessaurus is a carnivore at heart — raw, chaotic, and dangerous in the best way. I’m an herbivore, so don’t worry, nothing is getting eaten except plants, and the occasional bugs in the code. Right now it’s closer to a wild predator than a tamed framework — but that’s part of the fun.

If you want clean and polished, wait for the birds to evolve.
If you want teeth and claws… Clone Harnessaurus and join the hunt. 🦖

---
