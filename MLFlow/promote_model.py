import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = mlflow.MlflowClient()

model_name = "Iris_Classification_Model"
best_version = "2"

# Promote Version 2 as the champion model
client.set_registered_model_alias(
    model_name,
    "champion",
    best_version
)

# Mark older version as archived
client.set_model_version_tag(
    model_name,
    "1",
    "status",
    "archived"
)

print("----------------------------------------")
print("Model Promotion Completed")
print("----------------------------------------")
print("Model:", model_name)
print("Champion Version:", best_version)
print("Older Version 1: archived")
print("----------------------------------------")