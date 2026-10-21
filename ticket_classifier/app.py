"""Part 2 - Task 2: FastAPI service for the support ticket classifier."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

BASE = Path(__file__).parent
models = {}   # filled ONCE at startup, reused for every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    models["category"] = joblib.load(BASE / "baseline_category.joblib")
    models["priority"] = joblib.load(BASE / "baseline_priority.joblib")
    print("Models loaded")
    yield
    models.clear()


app = FastAPI(title="Support Ticket Classifier", version="1.0", lifespan=lifespan)


class Ticket(BaseModel):
    # strip spaces, then require at least 1 character -> "" or "   " returns 422
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=5000)]


class Prediction(BaseModel):
    category: str
    priority: str
    confidence: float            # confidence of the category prediction
    priority_confidence: float


def clean(text: str) -> str:
    return " ".join(text.lower().split())        # same cleaning as training


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(models) == 2}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    x = [clean(ticket.text)]
    cat_proba = models["category"].predict_proba(x)[0]
    pri_proba = models["priority"].predict_proba(x)[0]
    ci, pi = cat_proba.argmax(), pri_proba.argmax()
    return Prediction(
        category=str(models["category"].classes_[ci]),
        priority=str(models["priority"].classes_[pi]).capitalize(),
        confidence=round(float(cat_proba[ci]), 3),
        priority_confidence=round(float(pri_proba[pi]), 3),
    )