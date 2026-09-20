import mlflow

# MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Best run ID
run_id = "6acccae68349430180410d57591ab307"

# Model artifact logged in the run
model_uri = f"runs:/{run_id}/iris_random_forest_model"

# Registered model name
model_name = "Iris_Classification_Model"

# Register the best model
result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("----------------------------------------")
print("Model Registration Completed")
print("----------------------------------------")
print("Model Name:", result.name)
print("Model Version:", result.version)
print("----------------------------------------")