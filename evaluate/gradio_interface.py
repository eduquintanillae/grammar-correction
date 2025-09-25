import pandas as pd
import gradio as gr


def load_results(file_path):
    df = pd.read_csv(file_path)
    return df


def get_sentence_comparison(df1, df2, index):
    if index < 0 or index >= len(df1):
        return "Index out of range", "", "", "", "", ""

    row1 = df1.iloc[index]
    row2 = df2.iloc[index]

    return (
        row1["incorrect_sentence"],
        row1["correct_sentence"],
        row1["predicted_sentence"],
        row2["predicted_sentence"],
        row1["gleu"],
        row1["errant_f_score"],
        row2["gleu"],
        row2["errant_f_score"],
    )


def create_interface(df1, df2):
    # Get model names from the data
    model1_name = (
        df1["model_name"].iloc[0] if "model_name" in df1.columns else "Model 1"
    )
    model2_name = (
        df2["model_name"].iloc[0] if "model_name" in df2.columns else "Model 2"
    )

    with gr.Blocks() as demo:
        gr.Markdown("# Grammar Correction Model Comparison")
        gr.Markdown(f"**Model 1:** {model1_name} | **Model 2:** {model2_name}")
        gr.Markdown(f"Total sentences: {len(df1)}")

        with gr.Row():
            index_input = gr.Number(
                label="Sentence Index",
                value=0,
                precision=0,
                minimum=0,
                maximum=len(df1) - 1,
            )
            with gr.Column():
                prev_btn = gr.Button("← Previous", size="sm")
                next_btn = gr.Button("Next →", size="sm")

        with gr.Row():
            with gr.Column():
                incorrect_sentence = gr.Textbox(
                    label="Incorrect Sentence", lines=3, interactive=False
                )
            with gr.Column():
                correct_sentence = gr.Textbox(
                    label="Correct Sentence", lines=3, interactive=False
                )

        with gr.Row():
            with gr.Column():
                model1_prediction = gr.Textbox(
                    label=f"{model1_name} Prediction", lines=3, interactive=False
                )
                model1_metrics = gr.Textbox(
                    label=f"{model1_name} Metrics", lines=1, interactive=False
                )
            with gr.Column():
                model2_prediction = gr.Textbox(
                    label=f"{model2_name} Prediction", lines=3, interactive=False
                )
                model2_metrics = gr.Textbox(
                    label=f"{model2_name} Metrics", lines=1, interactive=False
                )

        def update(index):
            index = max(
                0, min(int(index), len(df1) - 1)
            )  # Ensure index is within bounds
            inc_sent, cor_sent, pred1, pred2, gleu1, f1_1, gleu2, f1_2 = (
                get_sentence_comparison(df1, df2, index)
            )
            model1_metrics_str = f"GLEU: {gleu1:.4f}, ERRANT F1: {f1_1:.4f}"
            model2_metrics_str = f"GLEU: {gleu2:.4f}, ERRANT F1: {f1_2:.4f}"
            return (
                index,
                inc_sent,
                cor_sent,
                pred1,
                pred2,
                model1_metrics_str,
                model2_metrics_str,
            )

        def prev_sentence(current_index):
            new_index = max(0, current_index - 1)
            return update(new_index)

        def next_sentence(current_index):
            new_index = min(len(df1) - 1, current_index + 1)
            return update(new_index)

        # Initialize with first sentence
        demo.load(
            fn=lambda: update(0),
            outputs=[
                index_input,
                incorrect_sentence,
                correct_sentence,
                model1_prediction,
                model2_prediction,
                model1_metrics,
                model2_metrics,
            ],
        )

        index_input.change(
            fn=update,
            inputs=index_input,
            outputs=[
                index_input,
                incorrect_sentence,
                correct_sentence,
                model1_prediction,
                model2_prediction,
                model1_metrics,
                model2_metrics,
            ],
        )

        prev_btn.click(
            fn=prev_sentence,
            inputs=index_input,
            outputs=[
                index_input,
                incorrect_sentence,
                correct_sentence,
                model1_prediction,
                model2_prediction,
                model1_metrics,
                model2_metrics,
            ],
        )

        next_btn.click(
            fn=next_sentence,
            inputs=index_input,
            outputs=[
                index_input,
                incorrect_sentence,
                correct_sentence,
                model1_prediction,
                model2_prediction,
                model1_metrics,
                model2_metrics,
            ],
        )

    return demo


if __name__ == "__main__":
    model1_results_path = "data/evaluation_results_gpt-4.1-mini_20250924_222853.csv"
    model2_results_path = (
        "data/evaluation_results_t5-grammar-corrector-3_20250925_040943.csv"
    )

    df1 = load_results(model1_results_path)
    df2 = load_results(model2_results_path)

    interface = create_interface(df1, df2)
    interface.launch()
