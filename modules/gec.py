import errant
from llm import LLM


class GrammarErrorCorrector:
    def __init__(self, model_name):
        self.llm = LLM(model_name)

    def correct(self, text):
        prompt = f"Correct the following sentence for grammar errors:\n\n{text}\n\nCorrected sentence:"
        system_prompt = "You are a helpful assistant that corrects grammar mistakes."
        response = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return response
