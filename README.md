# Grammar Correction

This project focuses on building an **Automatic Grammatical Error Correction (GEC)** system for English medical texts.

It involves comparing two approaches:
1. An out-of-the-box LLM (e.g., ChatGPT or Claude) using prompt engineering techniques.
2. A smaller model (e.g., T5-small from Google) fine-tuned.

The specific instructions for the challenge are available in the [Challenge.md](Challenge.md) (in Spanish).

### Installation

Create virtual environment and install dependencies:
```bash
python -m venv venv

venv\Scripts\activate  # On Windows
source venv/bin/activate  # On macOS/Linux

pip install -r requirements.txt
```

Make sure to also download the necessary spaCy model for evaluation:
```bash
python -m spacy download en_core_web_sm
```