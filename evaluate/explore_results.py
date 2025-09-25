import pandas as pd

if __name__ == "__main__":
    chatgpt_results_path = "data/evaluation_results_gpt-4.1-mini_20250924_222853.csv"
    t5_results_path = (
        "data/evaluation_results_t5-grammar-corrector-3_20250925_040943.csv"
    )

    chatgpt_results_df = pd.read_csv(chatgpt_results_path)
    print("\n--- ChatGPT results ---")
    print(chatgpt_results_df.describe())
    print(f"\nMean GLEU: {chatgpt_results_df['gleu'].mean()}")
    print(f"Mean ERRANT F1 Score: {chatgpt_results_df['errant_f_score'].mean()}")
    print(f"Mean time: {chatgpt_results_df['total_time'].mean()} seconds")

    t5_results_df = pd.read_csv(t5_results_path)
    print("\n--- T5 results ---")
    print(t5_results_df.describe())
    print(f"\nMean GLEU: {t5_results_df['gleu'].mean()}")
    print(f"Mean ERRANT F1 Score: {t5_results_df['errant_f_score'].mean()}")
    print(f"Mean time: {t5_results_df['total_time'].mean()} seconds")
