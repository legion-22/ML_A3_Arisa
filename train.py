import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.pyfunc
from LogisticRegression import LogisticRegression

# === Load data and preprocessin ===
df = pd.read_csv("Cars.csv")

owner_mapping = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 5
}
df["owner"] = df["owner"].map(owner_mapping)

df = df[~df["fuel"].isin(["CNG", "LPG"])]

df["mileage"] = df["mileage"].str.split(" ").str[0].astype(float)
df["engine"] = df["engine"].str.split(" ").str[0].astype(float)
df["max_power"] = df["max_power"].str.split(" ").str[0].astype(float)

df = df.drop(columns=[
    'name', 'year', 'fuel', 'seller_type', 'transmission', 'torque', 'seats'
])

df['selling_price'] = pd.qcut(df['selling_price'], q=4, labels=[0, 1, 2, 3])

df = df[df["owner"] != 5]

X = df[['km_driven', 'owner', 'mileage', 'engine', 'max_power']]
y = df['selling_price'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

for col in ['mileage', 'engine', 'max_power']:
    X_train[col].fillna(X_train[col].median(), inplace=True)
    X_test[col].fillna(X_train[col].median(), inplace=True)

features_to_scale = ['km_driven', 'mileage', 'engine', 'max_power']
scaler = StandardScaler()
X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
joblib.dump(scaler, 'scaler.pkl')

k = len(y.unique())
Y_train_encoded = np.zeros((y_train.shape[0], k))
for i, label in enumerate(y_train):
    Y_train_encoded[i, label] = 1

# === Train model ===
model = LogisticRegression(
    k=k,
    n=X_train.shape[1],
    method="sto",
    alpha=0.01,
    max_iter=5000,
    use_penalty=False,
    lambda_=0
) 
model.fit(X_train.values, Y_train_encoded)
yhat = model.predict(X_test.values)

class LogisticRegressionWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.values
        return self.model.predict(model_input)

# === Log to MLflow ===
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th/")
mlflow.set_experiment("st124879-a3")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'

with mlflow.start_run(run_name="final_model_log"):
    mlflow.log_param("method", model.method)
    mlflow.log_param("alpha", model.alpha)
    mlflow.log_param("lambda", model.lambda_)
    mlflow.log_param("use_penalty", model.use_penalty)
    mlflow.log_metric("accuracy", model.accuracy(y_test.values, yhat))

    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=LogisticRegressionWrapper(model),
        registered_model_name="st124879-a3-model"
    )

print("✅ Training complete and model logged to MLflow.")