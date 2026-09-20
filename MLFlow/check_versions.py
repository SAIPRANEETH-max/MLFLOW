import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = mlflow.MlflowClient()

model_name = "Iris_Classification_Model"

versions = client.search_model_versions(
    f"name='{model_name}'"
)

print("----------------------------------------")
print("Registered Model Versions")
print("----------------------------------------")

if not versions:
    print("No registered versions found.")
else:
    for version in versions:
        print("Version:", version.version)
        print("Run ID:", version.run_id)
        print("----------------------------------------")