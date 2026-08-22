import os
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score

DATA_DIR = "tourism_project/data"
MODEL_DIR = "tourism_project/deployment"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")


def load_data():
    Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
    Xtest = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))
    ytrain = pd.read_csv(os.path.join(DATA_DIR, "ytrain.csv")).iloc[:, 0]
    ytest = pd.read_csv(os.path.join(DATA_DIR, "ytest.csv")).iloc[:, 0]
    return Xtrain, Xtest, ytrain, ytest


def train_model():
    Xtrain, Xtest, ytrain, ytest = load_data()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }

    mlflow.set_experiment("Tourism_Package_Prediction")

    with mlflow.start_run():
        print("Step 1: Tuning hyperparameters with GridSearchCV...")
        base_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=(ytrain == 0).sum() / (ytrain == 1).sum(),
        )
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        print(f"Best params: {grid_search.best_params_}")
        mlflow.log_params(grid_search.best_params_)

        print("Step 2: Evaluating best model on train and test sets...")
        y_train_pred = best_model.predict(Xtrain)
        y_test_pred = best_model.predict(Xtest)

        train_report = classification_report(ytrain, y_train_pred, output_dict=True)
        test_report = classification_report(ytest, y_test_pred, output_dict=True)
        print("Test set report:")
        print(classification_report(ytest, y_test_pred))

        metrics = {
            "train_accuracy": accuracy_score(ytrain, y_train_pred),
            "train_f1_score": f1_score(ytrain, y_train_pred),
            "train_precision_1": train_report["1"]["precision"],
            "train_recall_1": train_report["1"]["recall"],
            "test_accuracy": accuracy_score(ytest, y_test_pred),
            "test_f1_score": f1_score(ytest, y_test_pred),
            "test_precision_1": test_report["1"]["precision"],
            "test_recall_1": test_report["1"]["recall"],
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        print("Step 3: Saving best model for deployment...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

    print("Training complete.")


if __name__ == "_main_":
    train_model()
