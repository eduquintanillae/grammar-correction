from .llm import LLM

SYSTEM_PROMPT = """You are an expert grammar correction assistant. Your task is to identify and correct grammatical errors, spelling mistakes, punctuation errors, and improve sentence structure while preserving the original meaning and tone.

Guidelines:
- Fix grammatical errors (subject-verb agreement, tense consistency, etc.)
- Correct spelling and punctuation mistakes
- Improve sentence structure and clarity when needed
- Maintain the original meaning, tone, and style
- Only make necessary corrections; don't rewrite unnecessarily
- Return only the corrected text without explanations unless specifically asked

Example:
Text: he go to school every day.
Corrected version: He goes to school every day."""

USER_PROMPT = """Please correct any grammatical errors, spelling mistakes, and punctuation issues in the following text while preserving its original meaning and tone:

Text: {text}

Corrected version:"""


class GrammarErrorCorrector:
    def __init__(self, model_name, system_prompt=SYSTEM_PROMPT, prompt=USER_PROMPT):
        self.model_name = model_name
        self.llm = LLM(model_name)
        self.system_prompt = system_prompt
        self.user_prompt = prompt

    def correct(self, text):
        prompt = self.user_prompt.format(text=text)
        response = self.llm.generate(prompt=prompt, system_prompt=self.system_prompt)
        return response
