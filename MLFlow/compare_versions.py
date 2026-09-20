import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Version 1
model_v1 = mlflow.pyfunc.load_model(
    "models:/Iris_Classification_Model/1"
)

# Load Version 2
model_v2 = mlflow.pyfunc.load_model(
    "models:/Iris_Classification_Model/2"
)

# Unseen data
new_data = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

prediction_v1 = model_v1.predict(new_data)
prediction_v2 = model_v2.predict(new_data)

print("----------------------------------------")
print("Model Version Comparison")
print("----------------------------------------")
print("Version 1 Prediction:", prediction_v1)
print("Version 2 Prediction:", prediction_v2)
print("----------------------------------------")