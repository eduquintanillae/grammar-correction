from .llm import LLM

SYSTEM_PROMPT = "You are a helpful assistant that corrects grammar mistakes."
USER_PROMPT = "Correct the following sentence: '{text}'"


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
