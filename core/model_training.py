from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV


def detect_problem_type(y):
    if y.dtype == "object":
        return "classification"
    if len(set(y)) <= 15:
        return "classification"
    return "regression"


def get_models(problem_type):
    if problem_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        param_grids = {
            "Random Forest": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        }

    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(),
            "SVR": SVR()
        }

        param_grids = {
            "Random Forest Regressor": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        }

    return models, param_grids