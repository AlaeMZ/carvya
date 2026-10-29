"""
app/main.py

FastAPI service for Moroccan used-car price estimation.

Design decisions carried over from the project notes:
  - Returns a RANGE plus a confidence flag, never a bare point estimate.
    "147,382 DH" implies a precision the model does not have (median
    error is ~13.5%).
  - The range widens automatically when the car cannot be identified,
    because measured out-of-fold error is roughly twice as large for
    those rows (p68 33.8% vs 16.5%).
  - The model artifact is loaded once at startup, not per request.

Run locally:
    uv run uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).parent / "static"

sys.path.append("src")
sys.path.append("src/models")

from predict import load_artifact, predict_price


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the artifact at startup rather than on the first request, so the
    first user does not eat the load latency and a missing/corrupt model
    file fails loudly at boot instead of silently at request time.
    """
    load_artifact()
    yield


app = FastAPI(
    title="Moroccan Used Car Price Estimator",
    description=(
        "Estimates the listing price of a used car in Morocco. "
        "Trained on publicly available Moroccan car listings, for academic research. "
        "[Contact me](https://github.com/AlaeMZ/carvya/issues) for any removal request. "
        "Returns a range, not a point estimate. "
        "Note: trained on ASKING prices, which typically run above "
        "final transaction prices."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    year: int = Field(..., ge=1970, le=2027, description="Model year")
    mileage_km: float | None = Field(
        None, ge=0, le=1_000_000, description="Odometer reading in km"
    )
    fuel: str | None = Field(
        None, description="Diesel, Essence, Hybride, Electrique, Lpg"
    )
    transmission: str | None = Field(
        None, description="Manuelle or Automatique"
    )
    city: str | None = Field(None, description="e.g. Casablanca, Rabat")
    title: str | None = Field(
        None,
        description=(
            "Listing title. Brand and model are extracted from this, so "
            "providing it substantially narrows the predicted range."
        ),
    )
    is_dealer: int = Field(0, ge=0, le=1)
    n_photos: int = Field(5, ge=0, le=50)

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "year": 2018,
                "mileage_km": 120000,
                "fuel": "Diesel",
                "transmission": "Manuelle",
                "city": "Casablanca",
                "title": "Dacia Logan Diesel Manuelle 2018 à Casablanca",
                "is_dealer": 0,
                "n_photos": 8,
            }]
        }
    }


class ExplanationItem(BaseModel):
    feature: str
    direction: str
    magnitude: float


class PredictionResponse(BaseModel):
    estimate_mad: float
    range_low_mad: float
    range_high_mad: float
    confidence: str
    detected_brand: str | None
    detected_model: str | None
    typical_error_pct: float
    explanation: list[ExplanationItem]


@app.get("/health")
def health() -> dict:
    try:
        art = load_artifact()
        return {
            "status": "ok",
            "n_training_rows": art["n_training_rows"],
            "median_error_pct": round(
                art["residual_quantiles"]["overall"]["p50"], 1
            ),
        }
    # any failure to load the artifact means the service is not ready
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest) -> dict:
    try:
        return predict_price(
            year=req.year,
            mileage_km=req.mileage_km,
            fuel=req.fuel,
            transmission=req.transmission,
            city=req.city,
            title=req.title,
            is_dealer=req.is_dealer,
            n_photos=req.n_photos,
        )
    # return a JSON error to the frontend instead of a bare 500 page
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/en", include_in_schema=False)
def root_en() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.en.html")