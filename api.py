from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
import traceback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
import math
import pandas as pd
from pandas.api import types as pdtypes

app = FastAPI(title="AI Research Assistant API")

model = None
best_model_name = None
dataset_name = None
feature_names = None
training_results = None

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    global model, best_model_name, dataset_name, feature_names, training_results
    print(f"/train called - filename={getattr(file, 'filename', None)} target_column={target_column}")

    # Read uploaded CSV file
    try:
        df = pd.read_csv(file.file)
        print(f"CSV loaded, columns={list(df.columns)}")
    except Exception as e:
        print("Error reading CSV:", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": f"Error reading CSV: {str(e)}"}, status_code=400)

    if target_column not in df.columns:
        return JSONResponse(
            content={"error": "Target column not found"},
            status_code=400
        )

    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X = pd.get_dummies(X)

        if y.dtype == "object":
            le = LabelEncoder()
            y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except Exception as e:
        print("Error during preprocessing/splitting:", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": f"Preprocessing error: {str(e)}"}, status_code=400)

    # Determine if this is a classification or regression problem
    problem_type = "classification"
    # If y is numeric and has many unique values, treat as regression
    try:
        if pdtypes.is_numeric_dtype(y):
            unique_count = int(pd.Series(y).nunique())
            if unique_count > 20:
                problem_type = "regression"
        else:
            problem_type = "classification"
    except Exception:
        problem_type = "classification"

    results = {}

    try:
        if problem_type == "classification":
            models = {
                "Random Forest": RandomForestClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "SVM": SVC()
            }

            for name, clf in models.items():
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, predictions)
                results[name] = round(acc, 4)

            # best model by accuracy
            best_metric_name = max(results, key=results.get)
            best_metric_value = results[best_metric_name]

        else:
            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "SVR": SVR()
            }

            for name, clf in models.items():
                clf.fit(X_train, y_train)
                predictions = clf.predict(X_test)
                # use R^2 as the comparable score (higher is better)
                r2 = r2_score(y_test, predictions)
                # handle NaN or infinite
                if r2 is None or (isinstance(r2, float) and (math.isnan(r2) or math.isinf(r2))):
                    r2 = -1.0
                results[name] = round(r2, 4)

            # best model by R^2
            best_metric_name = max(results, key=results.get)
            best_metric_value = results[best_metric_name]

    except Exception as e:
        print("Error during model training/evaluation:", str(e))
        traceback.print_exc()
        return JSONResponse(content={"error": f"Training error: {str(e)}"}, status_code=500)

    # Save metadata for report
    feature_names = list(X.columns)
    dataset_name = getattr(file, "filename", "uploaded_dataset.csv")

    # Keep the selected model object (from the chosen models dict)
    try:
        model = models[best_metric_name]
    except Exception:
        model = None

    # Prepare leaderboard (sorted list of tuples)
    leaderboard = sorted(results.items(), key=lambda x: x[1], reverse=True)

    training_results = {
        "problem_type": problem_type,
        "accuracies": results,
        "best_model": best_metric_name,
        "best_accuracy": best_metric_value,
        "leaderboard": leaderboard
    }

    return {
        "message": "Model comparison completed",
        "problem_type": problem_type,
        "accuracies": results,
        "best_model": best_metric_name,
        "best_accuracy": best_metric_value
    }


@app.get("/download-report")
def download_report():
    global training_results, dataset_name, feature_names

    if training_results is None:
        return JSONResponse(content={"error": "Generate a report first by training a model"}, status_code=400)

    try:
        filename = "temp_report.pdf"
        # Build leaderboard for PDF generator: list of (model, score)
        leaderboard = training_results.get("leaderboard", [])

        from core.report_generator import generate_pdf_report

        generate_pdf_report(
            filename=filename,
            dataset_name=dataset_name or "uploaded_dataset",
            problem_type="Classification",
            leaderboard=leaderboard,
            best_model=training_results.get("best_model"),
            best_score=training_results.get("best_accuracy"),
            feature_names=feature_names or []
        )

        return FileResponse(path=filename, media_type="application/pdf", filename="AI_Report.pdf")
    except Exception as e:
        return JSONResponse(content={"error": f"Could not generate report: {str(e)}"}, status_code=500)