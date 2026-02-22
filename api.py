from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

app = FastAPI(title="AI Research Assistant API")

model = None
best_model_name = None

@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    global model, best_model_name

    df = pd.read_csv(file.file)

    if target_column not in df.columns:
        return JSONResponse(
            content={"error": "Target column not found"},
            status_code=400
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X = pd.get_dummies(X)

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = {}

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        results[name] = round(acc, 4)

    # Select Best Model
    best_model_name = max(results, key=results.get)
    model = models[best_model_name]

    return {
        "message": "Model comparison completed",
        "accuracies": results,
        "best_model": best_model_name,
        "best_accuracy": results[best_model_name]
    }