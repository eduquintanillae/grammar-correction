from openai import OpenAI
import time
import os
import dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

dotenv.load_dotenv()


class LLM:
    def __init__(self, model_name):
        self.model_name = model_name

        if self.model_name in ["gpt-4.1-mini", "gpt-5-mini"]:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif "t5" in self.model_name:
            if torch.cuda.is_available():
                print("Using GPU for T5 model")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            else:
                raise EnvironmentError(
                    "GPU not available for T5 model. Please use a CPU-compatible model."
                )

    def generate(self, text, prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        prompt = prompt.format(text=text)
        messages.append({"role": "user", "content": prompt})

        initial_time = time.time()
        if self.model_name == "gpt-4.1-mini":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1500,
                stream=False,
            )
            model_response = response.choices[0].message.content.strip()
        elif self.model_name == "gpt-5-mini":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_completion_tokens=1500,
                stream=False,
            )
            model_response = response.choices[0].message.content.strip()
        elif "t5" in self.model_name:
            input_text = f"grammar: {text}"
            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt", max_length=256, truncation=True
            )

            output = self.model.generate(
                input_ids,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                repetition_penalty=2.5,
            )
            model_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        total_time = time.time() - initial_time

        results = {
            "model_response": model_response,
            "total_time": total_time,
        }

        return results
