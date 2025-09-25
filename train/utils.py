import pandas as pd

M2_EXTENSION = ".gold.bea19.m2"


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
        m2_path = f"./{key}{M2_EXTENSION}"
        sentences, edits = parse_m2_file(m2_path)
        temp_df = pd.DataFrame({"text": sentences, "edits": edits})
        temp_df["level"] = key.split(".")[0]
        temp_df["type"] = key.split(".")[1]
        df = pd.concat([df, temp_df], ignore_index=True)

    return df.reset_index().rename(columns={"index": "id"})


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
