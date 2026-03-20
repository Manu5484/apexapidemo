from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import numpy as np
from sklearn.linear_model import LogisticRegression

app = FastAPI(title="Project Scoring API")

# ---- Simple model trained at startup (demo) ----
# Features: budget, duration_days, team_size
# Target: high_risk (risk_score >= 4)
X_train = np.array([
    [50000, 120, 5],
    [75000, 150, 8],
    [60000,  90, 6],
    [90000, 200,10],
    [120000,250,12],
    [30000,  60, 3],
    [45000, 110, 4],
    [80000, 180, 9],
], dtype=float)

y_train = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=int)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

API_KEY = "change-me"  # demo only; use secrets manager/OCI vault etc. in real use


class ProjectRow(BaseModel):
    project_id: Optional[int] = None
    project_name: Optional[str] = None
    budget: float
    duration_days: float
    team_size: float


class PredictRequest(BaseModel):
    rows: List[ProjectRow]


class PredictRowResult(BaseModel):
    project_id: Optional[int] = None
    project_name: Optional[str] = None
    risk_probability: float
    risk_label: str


class PredictResponse(BaseModel):
    results: List[PredictRowResult]


def label_from_prob(p: float) -> str:
    if p >= 0.75:
        return "HIGH"
    if p >= 0.45:
        return "MEDIUM"
    return "LOW"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, x_api_key: str = Header(default="")):
    # simple header auth
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    X = np.array([[r.budget, r.duration_days, r.team_size] for r in req.rows], dtype=float)
    probs = model.predict_proba(X)[:, 1]  # probability of high risk

    results = []
    for r, p in zip(req.rows, probs):
        results.append(PredictRowResult(
            project_id=r.project_id,
            project_name=r.project_name,
            risk_probability=float(round(p, 4)),
            risk_label=label_from_prob(float(p))
        ))

    return PredictResponse(results=results)

