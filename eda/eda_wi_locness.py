import pandas as pd
import os
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data/wi+locness")
JSON_DIR = os.path.join(DATA_DIR, "json")
M2_DIR = os.path.join(DATA_DIR, "m2")
JSON_EXTENSION = ".json"
M2_EXTENSION = ".gold.bea19.m2"


def get_text_statistics(text: str) -> dict:
    if not text:
        return {"char_count": 0, "word_count": 0, "sentence_count": 0}

    char_count = len(text)
    word_count = len(text.split())
    sentence_count = len([s for s in text.split(".") if s.strip()])

    return {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
    }


def load_json_files(keys):
    data = {}

    for key in keys:
        path = os.path.join(JSON_DIR, f"{key}{JSON_EXTENSION}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line.strip())
                    stats = get_text_statistics(entry["text"])
                    data[entry["id"]] = {
                        "id": entry["id"],
                        "level": key.split(".")[0],
                        "type": key.split(".")[1],
                        "text": entry["text"],
                        "edits": entry.get("edits", []),
                        "char_count": stats["char_count"],
                        "word_count": stats["word_count"],
                        "sentence_count": stats["sentence_count"],
                    }
    df = pd.DataFrame.from_dict(data, orient="index").reset_index(drop=True)

    return df


def get_level_type_distribution(df: pd.DataFrame):
    distribution = df.groupby(["level", "type"]).size().unstack(fill_value=0)
    distribution["Total"] = distribution.sum(axis=1)

    return distribution


def analyze_text_lengths(df: pd.DataFrame):
    results = []

    for level in df["level"].unique():
        level_df = df[df["level"] == level]

        for metric_name, column in [
            ("Chars", "char_count"),
            ("Words", "word_count"),
            ("Sentences", "sentence_count"),
        ]:
            results.append(
                {
                    "Level": level,
                    "Metric": metric_name,
                    "Mean": level_df[column].mean().round(1),
                    "Median": level_df[column].median(),
                    "Std": level_df[column].std().round(1),
                    "Min": level_df[column].min(),
                    "Max": level_df[column].max(),
                }
            )

    return pd.DataFrame(results)


def parse_m2_file(file_path):
    sentences = []
    edits = []
    current_sentence = ""
    current_edits = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("S "):
                    if current_sentence:
                        sentences.append(current_sentence)
                        edits.append(current_edits)
                    current_sentence = line[2:]
                    current_edits = []
                elif line.startswith("A "):
                    current_edits.append(line[2:])

        if current_sentence:
            sentences.append(current_sentence)
            edits.append(current_edits)

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")

    return sentences, edits


def load_m2_files(keys):
    df = pd.DataFrame(columns=["text", "edits", "level", "type"])
    for key in keys:
        m2_path = os.path.join(M2_DIR, f"{key}{M2_EXTENSION}")
        sentences, edits = parse_m2_file(m2_path)
        temp_df = pd.DataFrame({"text": sentences, "edits": edits})
        temp_df["level"] = key.split(".")[0]
        temp_df["type"] = key.split(".")[1]
        df = pd.concat([df, temp_df], ignore_index=True)

    return df.reset_index().rename(columns={"index": "id"})


def get_error_types(edits):
    error_types = Counter()
    for edit_list in edits:
        for edit in edit_list:
            # Example: "start end|||TYPE|||correction|||REQUIRED|||-NONE-|||0"
            parts = edit.split("|||")
            if len(parts) >= 2:
                error_type = parts[1]
                error_types[error_type] += 1

    total_errors = sum(error_types.values())

    top_20_errors = {}
    for error_type, count in error_types.most_common(20):  # Top 20 error types
        percentage = (count / total_errors) * 100
        top_20_errors[error_type] = {"count": count, "percentage": round(percentage, 2)}

    df_top_20_errors = pd.DataFrame.from_dict(top_20_errors, orient="index")
    df_top_20_errors.index.name = "Error Type"

    return df_top_20_errors


def get_corrected_texts_from_json(dataset_df: pd.DataFrame):
    corrected_texts = []

    for _, row in dataset_df.iterrows():
        original_text = row["text"]
        edits = row.get("edits", [])

        if not edits or len(edits) == 0:
            corrected_texts.append(original_text)
            continue

        corrected_text = apply_json_edits(original_text, edits)
        corrected_texts.append(corrected_text)

    return corrected_texts


def apply_json_edits(text: str, edits: list) -> str:
    if not edits or len(edits) == 0:
        return text

    # Use the first annotator's edits (index 0)
    # Format: [annotator_id, [[start, end, replacement], [start, end, replacement], ...]]
    if len(edits[0]) < 2:
        return text

    edit_list = edits[0][1]  # Get the list of edits from first annotator

    # Sort edits by start position in reverse order to avoid index shifting
    sorted_edits = sorted(edit_list, key=lambda x: x[0], reverse=True)

    corrected_text = text

    for edit in sorted_edits:
        if len(edit) < 3:
            continue

        start, end, replacement = edit[0], edit[1], edit[2]

        # Handle different replacement types
        if replacement is None or replacement == "":
            # Deletion: remove text from start to end
            corrected_text = corrected_text[:start] + corrected_text[end:]
        else:
            # Replacement or insertion
            corrected_text = corrected_text[:start] + replacement + corrected_text[end:]

    return corrected_text


def get_corrected_texts_from_m2(dataset_df: pd.DataFrame):
    corrected_texts = []

    for _, row in dataset_df.iterrows():
        text = row["text"]
        edits = row["edits"]

        if not edits:
            corrected_texts.append(text)
            continue

        text_tokens = text.split()
        offset = 0

        for edit in edits:
            parts = edit.split("|||")
            if len(parts) < 3:
                continue

            span = parts[0].strip()
            correction = parts[2].strip()

            try:
                start, end = map(int, span.split())
                start += offset
                end += offset

                correction_tokens = correction.split() if correction != "-NONE-" else []
                text_tokens[start:end] = correction_tokens

                offset += len(correction_tokens) - (end - start)

            except ValueError:
                continue

        corrected_text = " ".join(text_tokens)
        corrected_texts.append(corrected_text)

    return corrected_texts


if __name__ == "__main__":
    keys = ["A.train", "A.dev", "B.train", "B.dev", "C.train", "C.dev", "N.dev"]
    df = load_json_files(keys)
    corrected_texts = get_corrected_texts_from_json(df)
    df["corrected_text"] = corrected_texts
    print(df.shape)
    distribution_df = get_level_type_distribution(df)
    length_stats_df = analyze_text_lengths(df)
    print("--- JSON File Analysis ---")
    print(f"\nNumber of entries: {df.shape[0]}")
    print("\nDistribution by Level and Type:")
    print(distribution_df)
    print("\nText Length Statistics:")
    print(length_stats_df)

    m2_df = load_m2_files(keys)
    m2_distribution_df = get_level_type_distribution(m2_df)
    error_types_df = get_error_types(m2_df["edits"])
    m2_df["corrected_text"] = get_corrected_texts_from_m2(m2_df)
    print("\n\n--- M2 File Analysis ---")
    print(f"\nNumber of entries: {m2_df.shape[0]}")
    print("\nM2 File Distribution by Level and Type:")
    print(m2_distribution_df)
    print("\nTop 20 Error Types:")
    print(error_types_df)
