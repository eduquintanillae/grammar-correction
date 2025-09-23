from openai import OpenAI
import time
import os
import dotenv

dotenv.load_dotenv()


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name == "gpt-4.1-mini":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        initial_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=1500,
            stream=False,
        )
        total_time = time.time() - initial_time

        model_response = response.choices[0].message.content.strip()
        results = {
            "model_response": model_response,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "total_time": total_time,
        }

        return results
