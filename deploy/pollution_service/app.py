"""Pollution weather systems service (FastAPI)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Import algorithms from docs/pollution_systems/code
BASE_DIR = Path(__file__).resolve().parents[2]
ALGO_DIR = BASE_DIR / "docs" / "pollution_systems" / "code"
if str(ALGO_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ALGO_DIR))

from high_pressure import identify_high_pressure_regional  # noqa: E402
from frontal_system import extract_frontal_system  # noqa: E402
from westerly_trough import extract_westerly_trough  # noqa: E402
from low_level_flow import extract_low_level_flow  # noqa: E402
from atmospheric_stability import extract_atmospheric_stability  # noqa: E402
from utils import GridContext  # noqa: E402

app = FastAPI(title="Pollution Systems Service", version="1.0.0")


class GridBase(BaseModel):
    lat: List[float]
    lon: List[float]
    center_lat: float
    center_lon: float


class HighPressureRequest(GridBase):
    z500: List[List[float]]


class FrontalRequest(GridBase):
    t850: List[List[float]]
    t500: List[List[float]]
    t1000: Optional[List[List[float]]] = None
    u925: Optional[List[List[float]]] = None
    v925: Optional[List[List[float]]] = None


class WesterlyRequest(GridBase):
    z500: List[List[float]]
    u500: Optional[List[List[float]]] = None
    v500: Optional[List[List[float]]] = None
    u200: Optional[List[List[float]]] = None


class LowLevelRequest(BaseModel):
    u10: float
    v10: float
    lat: float
    lon: float


class StabilityRequest(BaseModel):
    t2m_c: float
    lat: float
    lon: float


def _np2(arr: List[List[float]]) -> np.ndarray:
    return np.asarray(arr, dtype=float)


def _ctx(lat: List[float], lon: List[float]) -> GridContext:
    return GridContext.from_lat_lon(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))


def _auth_or_raise(authorization: Optional[str], api_key: str) -> None:
    if not api_key:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.split(" ", 1)[1].strip()
    if token != api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/high-pressure")
def high_pressure(req: HighPressureRequest, authorization: Optional[str] = Header(None)) -> dict:
    api_key = __import__("os").environ.get("API_KEY", "")
    _auth_or_raise(authorization, api_key)
    ctx = _ctx(req.lat, req.lon)
    result = identify_high_pressure_regional(ctx, _np2(req.z500), req.center_lat, req.center_lon)
    return {"result": result}


@app.post("/frontal")
def frontal(req: FrontalRequest, authorization: Optional[str] = Header(None)) -> dict:
    api_key = __import__("os").environ.get("API_KEY", "")
    _auth_or_raise(authorization, api_key)
    ctx = _ctx(req.lat, req.lon)
    result = extract_frontal_system(
        ctx,
        _np2(req.t850),
        _np2(req.t500),
        _np2(req.t1000) if req.t1000 is not None else None,
        _np2(req.u925) if req.u925 is not None else None,
        _np2(req.v925) if req.v925 is not None else None,
        req.center_lat,
        req.center_lon,
    )
    return {"result": result}


@app.post("/westerly-trough")
def westerly_trough(req: WesterlyRequest, authorization: Optional[str] = Header(None)) -> dict:
    api_key = __import__("os").environ.get("API_KEY", "")
    _auth_or_raise(authorization, api_key)
    ctx = _ctx(req.lat, req.lon)
    result = extract_westerly_trough(
        ctx,
        _np2(req.z500),
        _np2(req.u500) if req.u500 is not None else None,
        _np2(req.v500) if req.v500 is not None else None,
        _np2(req.u200) if req.u200 is not None else None,
        req.center_lat,
        req.center_lon,
    )
    return {"result": result}


@app.post("/low-level-flow")
def low_level_flow(req: LowLevelRequest, authorization: Optional[str] = Header(None)) -> dict:
    api_key = __import__("os").environ.get("API_KEY", "")
    _auth_or_raise(authorization, api_key)
    result = extract_low_level_flow(req.u10, req.v10, req.lat, req.lon)
    return {"result": result}


@app.post("/stability")
def stability(req: StabilityRequest, authorization: Optional[str] = Header(None)) -> dict:
    api_key = __import__("os").environ.get("API_KEY", "")
    _auth_or_raise(authorization, api_key)
    result = extract_atmospheric_stability(req.t2m_c, req.lat, req.lon)
    return {"result": result}
