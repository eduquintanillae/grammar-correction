from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import Dataset
import numpy as np
import torch
from .utils import load_m2_files, get_corrected_texts_from_m2
import nltk

PREFIX = "grammar: "


class Trainer:
    def __init__(
        self,
        model_name,
        max_length,
        output_dir,
        eval_strategy,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        weight_decay,
        save_total_limit,
        num_train_epochs,
        predict_with_generate,
        fp16,
        report_to,
    ):
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.max_length = max_length
        self.output_dir = output_dir
        self.eval_strategy = eval_strategy
        self.learning_rate = learning_rate
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.weight_decay = weight_decay
        self.save_total_limit = save_total_limit
        self.num_train_epochs = num_train_epochs
        self.predict_with_generate = predict_with_generate
        self.fp16 = fp16
        self.report_to = report_to

    def preprocess(self, example):
        input_text = PREFIX + example["text"]
        target_text = example["corrected_text"]
        model_inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        labels = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def correct_grammar(self, text: str):
        input_text = f"grammar: {text}"
        input_ids = self.tokenizer.encode(
            input_text, return_tensors="pt", truncation=True
        )

        input_ids = input_ids.to(self.model.device)

        output_ids = self.model.generate(input_ids, max_length=256, num_beams=4)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.split() for p in decoded_preds]
        decoded_labels = [l.split() for l in decoded_labels]

        scores = [
            nltk.translate.gleu_score.sentence_gleu([ref], hyp)
            for hyp, ref in zip(decoded_preds, decoded_labels)
        ]
        return {"gleu": 100 * sum(scores) / len(scores)}

    def train(self):
        keys = ["A.train", "A.dev"]
        dataset_df = load_m2_files(keys)
        corrected_texts = get_corrected_texts_from_m2(dataset_df)
        dataset_df["corrected_text"] = corrected_texts

        train_dataset = Dataset.from_pandas(
            dataset_df[dataset_df["type"] == "train"][["id", "text", "corrected_text"]]
        )
        val_dataset = Dataset.from_pandas(
            dataset_df[dataset_df["type"] == "dev"][["id", "text", "corrected_text"]]
        )
        tokenized_train = train_dataset.map(self.preprocess, batched=False)
        tokenized_val = val_dataset.map(self.preprocess, batched=False)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy=self.eval_strategy,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=self.save_total_limit,
            num_train_epochs=self.num_train_epochs,
            predict_with_generate=self.predict_with_generate,
            fp16=self.fp16,
            report_to=self.report_to,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


if __name__ == "__main__":
    model_name = "t5-small"

    trainer = Trainer(
        model_name,
        max_length=64,
        output_dir="./t5-grammar-corrector-3",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.0,
        save_total_limit=1,
        num_train_epochs=20,
        predict_with_generate=True,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
    )
    trainer.train()
