"""Diabetes risk API: wraps predict() behind a URL."""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(title="Diabetes Risk Screener")

# Lets a web page opened from your disk call this API (needed on Day 3's form)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Person(BaseModel):
    Age: int = Field(ge=1, le=13, description="Age bracket code 1-13")
    BMI: float = Field(ge=10, le=100)
    HighBP: int = Field(ge=0, le=1)
    HighChol: int = Field(ge=0, le=1)
    PhysActivity: int = Field(ge=0, le=1)
    GenHlth: int = Field(ge=1, le=5, description="1 excellent ... 5 poor")
    Sex: int = Field(ge=0, le=1, description="0 female, 1 male")
    Smoker: int = Field(ge=0, le=1)


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(person: Person):
    try:
        return predict(person.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))