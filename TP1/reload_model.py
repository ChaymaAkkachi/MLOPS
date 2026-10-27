
import mlflow
import mlflow.sklearn
import pandas as pd

# ===== 1. Find the best run =====
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

if runs.empty:
    raise ValueError(
        "No runs found. Please train the models first."
    )

run_id = runs.iloc[0]["run_id"]

# ===== 2. Reload the best model =====
model = mlflow.sklearn.load_model(
    f"runs:/{run_id}/model"
)

print("Model reloaded:", type(model).__name__)
print("Number of expected variables:", model.n_features_in_)

# ===== 3. Create a new customer =====
# Example: Standard subscription
customer = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# ===== 4. Keep the features expected by the model =====
# Use the same columns and order as during training
customer = customer.reindex(
    columns=model.feature_names_in_,
    fill_value=0
)

# ===== 5. Predict customer churn =====
prediction = model.predict(customer)[0]

print(
    "Will the customer leave? ->",
    "YES" if prediction == 1 else "NO"
)
