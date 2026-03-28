import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import joblib
import os

# Set MLflow tracking
mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_experiment("credit_risk_model")

# Generate synthetic credit risk data
def generate_credit_data(n_samples=5000):
    np.random.seed(42)
    
    income = np.random.normal(50000, 20000, n_samples)
    debt = np.random.normal(10000, 5000, n_samples)
    age = np.random.randint(18, 70, n_samples)
    credit_history = np.random.randint(300, 850, n_samples)
    employed_years = np.random.exponential(5, n_samples)
    
    # Risk score based on realistic patterns
    risk_score = (
        (debt / income) * 0.4 +
        ((850 - credit_history) / 550) * 0.4 +
        (np.maximum(0, 30 - age) / 30) * 0.2
    )
    risk_score += np.random.normal(0, 0.1, n_samples)
    default = (risk_score > 0.6).astype(int)
    
    return pd.DataFrame({
        'income': income,
        'debt': debt,
        'age': age,
        'credit_history': credit_history,
        'employed_years': employed_years,
        'default': default
    })

print("📊 Generating credit risk data...")
data = generate_credit_data(5000)

feature_cols = ['income', 'debt', 'age', 'credit_history', 'employed_years']
X = data[feature_cols]
y = data['default']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"📈 Training data: {len(X_train)} samples")
print(f"📊 Test data: {len(X_test)} samples")
print(f"🎯 Default rate: {y.mean()*100:.1f}%")

# Train model
print("🚀 Training Random Forest model...")
with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Log to MLflow
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    })
    mlflow.log_metrics({
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall
    })
    
    # Log feature importance
    for col, imp in zip(feature_cols, model.feature_importances_):
        mlflow.log_metric(f"importance_{col}", imp)
    
    # Log model
    mlflow.sklearn.log_model(model, "credit_risk_model")
    
    # Save locally
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f"models/credit_model.pkl")
    
    print(f"\n✅ Training Complete!")
    print(f"   Run ID: {run.info.run_id}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"\n🔗 View in MLflow: http://localhost:5001")
