from __future__ import annotations
from typing import Any, List, Dict, Optional
import pandas as pd
from autoguard.core.logging import get_logger

logger = get_logger(__name__)

# Must be at module level for Pydantic v2 to resolve correctly
try:
    from pydantic import BaseModel
    class RowData(BaseModel):
        data: List[Dict[str, Any]]
except ImportError:
    pass


def create_app(ag=None):
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="AutoGuard ML API", version="0.1.0",
                  description="AutoML + Drift Detection REST API")
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    _state = {"ag": ag}

    def _ag():
        inst = _state.get("ag")
        if inst is None or not getattr(inst, "is_fitted", False):
            raise HTTPException(status_code=503, detail="Model not loaded.")
        return inst

    @app.get("/health")
    async def health():
        inst = _state.get("ag")
        loaded = inst is not None and getattr(inst, "is_fitted", False)
        return {
            "status": "ok",
            "model_loaded": loaded,
            "best_model": inst.best_model_name if loaded else None,
            "problem_type": inst._problem_type if loaded else None,
            "version": "0.1.0",
        }

    @app.get("/model/info")
    async def model_info():
        inst = _ag()
        result = {
            "model_name": inst.best_model_name,
            "problem_type": inst._problem_type,
            "target": inst._target,
            "features": inst._feature_cols,
        }
        if inst.leaderboard is not None:
            result["leaderboard"] = inst.leaderboard.to_dict(orient="records")
        return result

    @app.post("/predict")
    async def predict(body: RowData):
        inst = _ag()
        try:
            preds = inst.predict(pd.DataFrame(body.data))
            return {"predictions": preds.tolist(),
                    "model_name": inst.best_model_name,
                    "n_samples": len(preds)}
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/predict/proba")
    async def predict_proba(body: RowData):
        inst = _ag()
        try:
            proba = inst.predict_proba(pd.DataFrame(body.data))
            return {"probabilities": proba.tolist(),
                    "classes": [str(i) for i in range(proba.shape[1])],
                    "n_samples": len(proba)}
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.post("/monitor")
    async def monitor(body: RowData):
        inst = _ag()
        try:
            r = inst.monitor(pd.DataFrame(body.data), save_report=False)
            return {"overall_drift_severity": r["overall_drift_severity"],
                    "drift_level": r["drift_level"],
                    "drifted_features": r["drifted_features"],
                    "timestamp": r["timestamp"]}
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app
