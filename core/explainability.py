import shap
import matplotlib.pyplot as plt


def generate_shap_explanation(model, X):
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        print("\n🔍 Generating SHAP summary plot...")
        shap.summary_plot(shap_values, X, show=True)

    except Exception as e:
        print("⚠️ SHAP explanation not supported for this model.")
        print("Reason:", e)