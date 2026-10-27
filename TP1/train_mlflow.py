
# ===== 1. Import the libraries =====
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ===== 2. Load the data (only once) =====
df = pd.read_csv("churn_data.csv")

y = df["churn"]  # 1 = customer leaves, 0 = customer stays


# ===== 3. Function: prepare the data for a list of columns =====
def prepare(columns):
    # Convert categorical columns into numerical columns
    X = pd.get_dummies(df[columns], drop_first=True)

    # 80% training / 20% testing
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


# ===== 4. Function: run ONE trial and record it in MLflow =====
def run_trial(name, n_estimators, max_depth, columns):

    X_train, X_test, y_train, y_test = prepare(columns)

    with mlflow.start_run(run_name=name):

        # Create the model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        pred = model.predict(X_test)

        # --- Parameters: settings of this trial ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("features", ",".join(columns))

        # --- Metrics: evaluate the model ---
        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(
            y_test, pred, zero_division=0
        )

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # --- Artifact: save the trained model in MLflow ---
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="cloudpickle"
        )

        # Display the results
        print(
            name,
            "-> accuracy:", round(accuracy, 3),
            "| precision:", round(precision, 3),
            "| recall:", round(recall, 3),
            "| f1:", round(f1, 3)
        )


# ===== 5. Main program =====

# Create or select the experiment
mlflow.set_experiment("churn-prediction")

# All available input features
ALL = [
    "age",
    "salaire",
    "abonnement",
    "anciennete",
    "consommation"
]

# Trial 1: 100 trees, maximum depth 5, all features
run_trial("trial-1", 100, 5, ALL)

# Trial 2: 50 trees, maximum depth 3, all features
run_trial("trial-2", 50, 3, ALL)

# Trial 3: 100 trees, maximum depth 5, without salary
run_trial(
    "trial-3",
    100,
    5,
    ["age", "abonnement", "anciennete", "consommation"]
)
