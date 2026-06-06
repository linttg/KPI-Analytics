from fastapi import FastAPI
import uvicorn
import numpy as np

from engine import StrategyEngine

app = FastAPI(
    title="Strategy AI API"
)

engine = StrategyEngine()


@app.get("/")
def home():

    return {
        "message": "API Online"
    }


@app.post("/predict")
def predict(payload: dict):

    matrix = np.array(payload["matrix"])

    weights, cr = engine.calculate_ahp(matrix)

    return {
        "weights": weights.tolist(),
        "consistency_ratio": float(cr),
        "is_consistent": bool(cr < 0.1)
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )