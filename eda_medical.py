import pandas as pd
import nltk
from llm import LLM

nltk.download("punkt")

GENERATE_EXPLANATIONS = False

df = pd.read_csv("data/data.csv")
print(f"Columns: {df.columns}, Shape: {df.shape}")
print(f"\nFirst Incorrect Sentence:\n{df.iloc[0]['incorrect_sentence']}")
print(f"\nFirst Correct Sentence:\n{df.iloc[0]['correct_sentence']}")


df["inc_sentence_tokens"] = df["incorrect_sentence"].apply(
    lambda s: len(nltk.word_tokenize(str(s)))
)
df["cor_sentence_tokens"] = df["correct_sentence"].apply(
    lambda s: len(nltk.word_tokenize(str(s)))
)
print(f"\nIncorrect Sentence Token Stats:\n{df['inc_sentence_tokens'].describe()}")
print(f"\nCorrect Sentence Token Stats:\n{df['cor_sentence_tokens'].describe()}")

if GENERATE_EXPLANATIONS:
    llm = LLM("gpt-4.1-mini")
    system_prompt = "You are a helpful assistant that recognizes the grammar mistake made between an incorrect sentence and the corrected sentence."

    mistake_explanations = []
    for i, row in df.iterrows():
        print(f"Processing row {i + 1}/{len(df.iloc[:2])}")
        prompt = f"Identify the grammar mistake in the following sentence with one to three words representing the main grammar topic:\n\nIncorrect Sentence: {row['incorrect_sentence']}\nCorrected Sentence: {row['correct_sentence']}\n\nMistake:"
        result = llm.generate(prompt, system_prompt=system_prompt)
        mistake_explanations.append(result["model_response"])
    df["mistake_explanation"] = mistake_explanations
    df["mistake_explanation"] = df["mistake_explanation"].apply(
        lambda x: str(x.replace("Mistake: ", "")).strip().lower()
    )
    df.to_csv("data/data_detailed.csv", index=False)


df = pd.read_csv("data/data_detailed.csv")
print(f"\nUnique Mistake Explanations:\n{df['mistake_explanation'].unique()}")
