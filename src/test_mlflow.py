import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Connect to MLflow
mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_experiment("credit_risk_test")

# Create simple data
X = np.random.rand(100, 4)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# Train and log
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("n_estimators", 10)
    mlflow.sklearn.log_model(model, "model")
    
    print(f"✅ Success! Experiment logged to MLflow")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   View at: http://localhost:5001")
