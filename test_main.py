import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_missing_phase_400():
    res = client.post("/quantize", json={})
    assert res.status_code == 400
    assert res.json() == {"error": "INVALID_INPUT"}

    res2 = client.post("/quantize", json={"phase": "unknown"})
    assert res2.status_code == 400
    assert res2.json() == {"error": "INVALID_INPUT"}

def test_freeze_and_select_success_flow():
    freeze_payload = {
        "phase": "freeze",
        "freezeId": "freeze_001",
        "calibrationDigest": "calib_123",
        "tokenizerDigest": "tok_456",
        "allowedUnsupportedReasons": ["REASON_OK"],
        "candidates": [
            {
                "name": "int4",
                "files": {"model.safetensors": "content_int4"},
                "loadable": True,
                "calibrationDigest": "calib_123",
                "tokenizerDigest": "tok_456"
            },
            {
                "name": "int8",
                "files": {"model.safetensors": "content_int8_larger"},
                "loadable": True,
                "calibrationDigest": "calib_123",
                "tokenizerDigest": "tok_456"
            }
        ]
    }

    res_f = client.post("/quantize", json=freeze_payload)
    assert res_f.status_code == 200
    f_data = res_f.json()
    assert f_data["freezeId"] == "freeze_001"
    assert len(f_data["candidates"]) == 2
    assert f_data["candidates"][0]["status"] == "frozen"
    assert f_data["candidates"][1]["status"] == "frozen"
    frozen_candidates = f_data["candidates"]

    # Replay freeze -> 200 OK
    replay_res = client.post("/quantize", json=freeze_payload)
    assert replay_res.status_code == 200

    # Conflict freeze -> 409
    modified_freeze = dict(freeze_payload)
    modified_freeze["calibrationDigest"] = "different_calib"
    conflict_res = client.post("/quantize", json=modified_freeze)
    assert conflict_res.status_code == 409
    assert conflict_res.json() == {"error": "FREEZE_ID_CONFLICT"}

    # Phase select
    select_payload = {
        "phase": "select",
        "freezeId": "freeze_001",
        "candidates": frozen_candidates,
        "policy": {
            "maxBytes": 1000000,
            "aggregateFloor": 0.8,
            "requiredSlices": {"critical": 0.75},
            "maxLatencyMs": 100,
            "candidateOrder": ["int4", "int8"]
        },
        "latencies": {"int4": 40, "int8": 60},
        "rows": [
            {"label": 1, "slice": "critical", "predictions": {"int4": 1, "int8": 1}},
            {"label": 0, "slice": "critical", "predictions": {"int4": 0, "int8": 0}}
        ]
    }

    res_s = client.post("/quantize", json=select_payload)
    assert res_s.status_code == 200
    s_data = res_s.json()
    assert s_data["selected"] == "int4"  # int4 has smaller bytes!
    assert s_data["packageManifest"]["name"] == "int4"
    assert len(s_data["results"]) == 2
    assert s_data["results"][0]["admitted"] is True
    assert s_data["results"][1]["admitted"] is True
