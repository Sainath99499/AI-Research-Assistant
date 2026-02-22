import pandas as pd
import os
import traceback

from core.data_cleaning import clean_data
from core.model_training import detect_problem_type, get_models
from core.evaluation import evaluate_models
from core.explainability import generate_shap_explanation
from core.report_generator import generate_pdf_report

def run_automl(file_path, target_column):
    # Ensure .csv extension
    if not file_path.endswith(".csv"):
        file_path += ".csv"

    # Check file existence
    if not os.path.exists(file_path):
        print(f"❌ Error: File '{file_path}' not found.")
        available = [f for f in os.listdir('.') if f.endswith('.csv')]
        if available:
            print(f"💡 Available CSV files in current folder: {available}")
        return

    # Try reading CSV
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return

    # Validate target column
    if target_column not in df.columns:
        print(f"❌ Error: Column '{target_column}' not found.")
        print(f"💡 Available columns: {df.columns.tolist()}")
        return

    try:
        print("\n🔍 Cleaning and preprocessing data...")
        X, y = clean_data(df, target_column)

        print("🧠 Detecting problem type...")
        problem_type = detect_problem_type(y)
        print(f"✔ Problem Type Detected: {problem_type}")

        print("⚙️ Training and tuning models...")
        models, param_grids = get_models(problem_type)

        results, best_model_name, best_model = evaluate_models(
            models, param_grids, X, y, problem_type
        )

        # Sort leaderboard
        leaderboard = sorted(results.items(), key=lambda x: x[1], reverse=True)

        print("\n📊 MODEL LEADERBOARD")
        print("=" * 40)
        for rank, (model, score) in enumerate(leaderboard, start=1):
            print(f"{rank}. {model} → {score}")

        print("\n🏆 Best Model:", best_model_name)
        print("📈 Best Score:", results[best_model_name])

        # SHAP Explainability
        print("\n🧠 Generating Model Explainability...")
        generate_shap_explanation(best_model, X)

        # Generate PDF Report
        print("\n📄 Generating PDF Report...")
        generate_pdf_report(
            filename="AutoML_Report.pdf",
            dataset_name=file_path,
            problem_type=problem_type,
            leaderboard=leaderboard,
            best_model=best_model_name,
            best_score=results[best_model_name],
            feature_names=X.columns.tolist()
        )

    except Exception as e:
        print("❌ Error during AutoML processing.")
        print("Details:", e)
        traceback.print_exc()


if __name__ == "__main__":
    print("🤖 AutoML AI Research Assistant")
    print("=" * 40)

    file_path = input("Enter CSV file path (example: 'iris'): ").strip()
    target_column = input("Enter target column name: ").strip()

    if file_path and target_column:
        run_automl(file_path, target_column)
    else:
        print("❌ File path and target column are required.")