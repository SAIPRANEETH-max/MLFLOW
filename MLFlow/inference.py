import mlflow
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the champion model
model = mlflow.pyfunc.load_model(
    "models:/Iris_Classification_Model@champion"
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

# Prediction
prediction = model.predict(new_data)

print("----------------------------------------")
print("Inference using Champion Model")
print("----------------------------------------")
print("Input:")
print(new_data)
print("Prediction:", prediction)
print("----------------------------------------")