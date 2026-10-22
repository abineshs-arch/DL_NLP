"""Week 3 Assessment - Task 4: FastAPI serving the fine-tuned DistilBERT sentiment model."""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints
from transformers import pipeline

MODEL_DIR = Path(__file__).parent / "sentiment_model"
state = {}   # the model is loaded ONCE here and reused for every request


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["clf"] = pipeline("text-classification", model=str(MODEL_DIR))
    print("Model loaded")
    yield
    state.clear()


app = FastAPI(title="Feedback Sentiment API", version="1.0", lifespan=lifespan)


class Feedback(BaseModel):
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class Prediction(BaseModel):
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "clf" in state}


@app.post("/predict", response_model=Prediction)
def predict(item: Feedback):
    result = state["clf"](item.text.lower())[0]
    return Prediction(label=result["label"], confidence=round(float(result["score"]), 3))