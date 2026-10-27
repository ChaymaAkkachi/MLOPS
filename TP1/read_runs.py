
import mlflow  # Reuse MLflow to read the results

# Search for all runs in the experiment and sort by F1 score
# DESC = descending order (best F1 score first)
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"]
)

# Columns to display
columns = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1"
]

# Display the results without row numbers
print(runs[columns].to_string(index=False))

# The first row is the best run (highest F1 score)
if not runs.empty:
    best = runs.iloc[0]

    print()

    # Display the best run's name and first 8 characters of its ID
    print(
        "Best run:",
        best["tags.mlflow.runName"],
        "| id:",
        best["run_id"][:8],
        "..."
    )
else:
    print("No runs found for the experiment 'churn-prediction'.")
