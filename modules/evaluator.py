from llm import LLM
import pandas as pd
from nltk.translate.gleu_score import sentence_gleu
import errant


class Evaluator:
    def __init__(self, model_name, dataset_path):
        self.llm = LLM(model_name)
        self.dataset = self.load_dataset(dataset_path)
        self.annotator = errant.load("en")

    def load_dataset(self, path):
        df = pd.read_csv(path)
        return df

    def predict(self, text):
        system_prompt = "You are a helpful assistant that corrects grammar mistakes."
        prompt = f"Correct the following sentence: '{text}'"
        prediction = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
        return prediction

    def evaluate_errant(self, incorrect_text, corrected_text, predicted_text, metrics):
        try:
            orig = self.annotator.parse(incorrect_text)
            ref = self.annotator.parse(corrected_text)
            hyp = self.annotator.parse(predicted_text)

            gold_edits = self.annotator.annotate(orig, ref)
            sys_edits = self.annotator.annotate(orig, hyp)

            tp = 0
            fp = 0
            fn = 0

            # Convert edits to comparable format
            gold_set = set(
                (edit.o_start, edit.o_end, edit.c_str) for edit in gold_edits
            )
            sys_set = set((edit.o_start, edit.o_end, edit.c_str) for edit in sys_edits)

            tp = len(gold_set & sys_set)
            fp = len(sys_set - gold_set)
            fn = len(gold_set - sys_set)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f_score = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            metrics["errant_precision"] = precision
            metrics["errant_recall"] = recall
            metrics["errant_f_score"] = f_score

        except Exception as e:
            print(f"ERRANT evaluation failed: {e}")
            metrics["errant_precision"] = 0
            metrics["errant_recall"] = 0
            metrics["errant_f_score"] = 0

        return metrics

    def evaluate(self, incorrect_text, corrected_text, predicted_text):
        metrics = {}
        reference_tokens = corrected_text.split()
        prediction_tokens = predicted_text.split()

        metrics["gleu"] = sentence_gleu([reference_tokens], prediction_tokens)
        metrics = self.evaluate_errant(
            incorrect_text, corrected_text, predicted_text, metrics
        )

        return metrics

    def run_evaluation(self):
        self.dataset["model_name"] = self.llm.model_name
        for i, row in self.dataset.iterrows():
            print(f"Evaluating row {i + 1}/{len(self.dataset)}")
            incorrect_text = row["incorrect_sentence"]
            corrected_text = row["correct_sentence"]
            prediction = self.predict(incorrect_text)
            self.dataset.at[i, "predicted_sentence"] = prediction["model_response"]
            self.dataset.at[i, "total_time"] = prediction["total_time"]
            evaluation = self.evaluate(
                incorrect_text, corrected_text, prediction["model_response"]
            )
            self.dataset.at[i, "gleu"] = evaluation["gleu"]
            self.dataset.at[i, "errant_precision"] = evaluation["errant_precision"]
            self.dataset.at[i, "errant_recall"] = evaluation["errant_recall"]
            self.dataset.at[i, "errant_f_score"] = evaluation["errant_f_score"]
        print(f"Evaluation complete for {len(self.dataset)} samples.")


if __name__ == "__main__":
    evaluator = Evaluator("gpt-5-mini", "data/data.csv")
    evaluator.run_evaluation()
    evaluator.dataset.to_csv("data/evaluation_results.csv", index=False)
