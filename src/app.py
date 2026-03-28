from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
from typing import Optional

app = FastAPI(title="Credit Risk Scoring API", version="1.0", description="Predict loan default risk for customers")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set MLflow tracking
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

# Global model variable
model = None
model_run_id = None

class CustomerData(BaseModel):
    income: float
    debt: float
    age: int
    credit_history: int
    employed_years: float

class RiskResponse(BaseModel):
    customer_id: Optional[str] = None
    risk_score: float
    risk_level: str
    default_probability: str
    recommendation: str
    model_version: str
    feature_importance: dict

@app.on_event("startup")
async def load_model():
    global model, model_run_id
    
    print("🔄 Loading credit risk model...")
    
    try:
        # Find the latest run from credit_risk_model experiment
        experiment = mlflow.get_experiment_by_name("credit_risk_model_v2")
        
        if experiment:
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                model_run_id = runs.iloc[0]['run_id']
                model_uri = f"runs:/{model_run_id}/credit_risk_model"
                model = mlflow.sklearn.load_model(model_uri)
                
                # Get feature importance from the run
                feature_importance = {}
                for col in ['income', 'debt', 'age', 'credit_history', 'employed_years']:
                    metric_name = f"importance_{col}"
                    if metric_name in runs.iloc[0].index:
                        feature_importance[col] = runs.iloc[0][metric_name]
                
                print(f"✅ Model loaded successfully!")
                print(f"   Run ID: {model_run_id}")
                print(f"   Feature Importance: {feature_importance}")
            else:
                print("⚠️ No model found in MLflow. Please run training first!")
        else:
            print("⚠️ Experiment 'credit_risk_model' not found. Run training first!")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

@app.get("/")
async def root():
    return {
        "service": "Credit Risk Scoring API",
        "status": "running",
        "model_loaded": model is not None,
        "model_version": model_run_id,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (GET or POST)",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_run_id": model_run_id,
        "service": "credit-risk-scoring"
    }

@app.post("/predict", response_model=RiskResponse)
async def predict_risk(customer: CustomerData):
    """Predict credit risk for a customer"""
    
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please run training first: docker-compose exec dev python src/train.py"
        )
    
    # Prepare features
    features = pd.DataFrame([[
        customer.income,
        customer.debt,
        customer.age,
        customer.credit_history,
        customer.employed_years
    ]], columns=['income', 'debt', 'age', 'credit_history', 'employed_years'])
    
    # Get prediction
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    
    risk_score = float(proba[1])
    print(f"DEBUG: Customer {customer.income}/{customer.debt}/{customer.age}/{customer.credit_history}/{customer.employed_years} -> Risk Score: {risk_score:.4f}")
    
    # Determine risk level and recommendation
    if risk_score < 0.3:
        risk_level = "🟢 LOW RISK"
        recommendation = "APPROVE LOAN - Customer has good credit profile"
        risk_description = "Low probability of default"
    elif risk_score < 0.6:
        risk_level = "🟡 MEDIUM RISK"
        recommendation = "REVIEW MANUALLY - Requires additional verification"
        risk_description = "Moderate probability of default"
    else:
        risk_level = "🔴 HIGH RISK"
        recommendation = "REJECT LOAN - Customer does not meet credit criteria"
        risk_description = "High probability of default"
    
    # Get feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        feature_names = ['income', 'debt', 'age', 'credit_history', 'employed_years']
        for name, imp in zip(feature_names, model.feature_importances_):
            feature_importance[name] = round(float(imp), 3)
    
    return RiskResponse(
        customer_id=None,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        default_probability=f"{risk_score*100:.1f}%",
        recommendation=recommendation,
        model_version=model_run_id[:8] if model_run_id else "unknown",
        feature_importance=feature_importance
    )

@app.get("/predict")
async def predict_get(
    income: float,
    debt: float,
    age: int,
    credit_history: int,
    employed_years: float
):
    """Predict credit risk using query parameters"""
    customer = CustomerData(
        income=income,
        debt=debt,
        age=age,
        credit_history=credit_history,
        employed_years=employed_years
    )
    return await predict_risk(customer)

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        return {"status": "no model loaded"}
    
    return {
        "model_version": model_run_id,
        "model_type": "RandomForestClassifier",
        "feature_importance": dict(zip(
            ['income', 'debt', 'age', 'credit_history', 'employed_years'],
            model.feature_importances_.round(3).tolist()
        )) if hasattr(model, 'feature_importances_') else None
    }
