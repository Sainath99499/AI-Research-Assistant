from sklearn.model_selection import cross_val_score, GridSearchCV
import numpy as np


def evaluate_models(models, param_grids, X, y, problem_type):
    results = {}
    trained_models = {}

    for name, model in models.items():

        if name in param_grids:
            grid = GridSearchCV(
                model,
                param_grids[name],
                cv=3,
                scoring="accuracy" if problem_type == "classification" else "r2"
            )
            grid.fit(X, y)
            best_model = grid.best_estimator_
            score = grid.best_score_
            trained_models[name] = best_model
        else:
            scores = cross_val_score(
                model,
                X,
                y,
                cv=5,
                scoring="accuracy" if problem_type == "classification" else "r2"
            )
            model.fit(X, y)
            trained_models[name] = model
            score = np.mean(scores)

        results[name] = round(score, 4)

    # Sort leaderboard
    leaderboard = sorted(results.items(), key=lambda x: x[1], reverse=True)

    best_model_name = leaderboard[0][0]
    best_model = trained_models[best_model_name]

    return results, best_model_name, best_model