
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===== 1. Load the data =====
df = pd.read_csv("churn_data.csv")

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# ===== 2. Prepare features and target =====
X = pd.get_dummies(
    df[["age", "salaire", "abonnement", "anciennete", "consommation"]],
    drop_first=True
)

y = df["churn"]

# ===== 3. Split the data =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 4. Configure MLflow =====
mlflow.set_experiment("churn-prediction")

mlflow.sklearn.autolog(
    serialization_format="cloudpickle",
    log_post_training_metrics=False
)

# ===== 5. Train and log the model =====
with mlflow.start_run(run_name="trial-auto"):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate the model on the test set
    predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, predictions)

    # Explicitly log test accuracy
    mlflow.log_metric("test_accuracy", test_accuracy)

    print("Test accuracy:", round(test_accuracy, 3))

print("Training completed successfully!")
