from modules.evaluator import Evaluator
import pandas as pd

if __name__ == "__main__":
    model_name = "./t5-grammar-corrector-3"
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    evaluator = Evaluator(model_name, "data.csv")
    evaluator.run_evaluation()
    if "./" in model_name:
        model_name = model_name.split("./")[1]
    evaluator.dataset.to_csv(
        f"./evaluation_results_{model_name}_{timestamp}.csv", index=False
    )
