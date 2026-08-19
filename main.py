import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Set
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Quantized Model Candidate Admission Service")

STORE: Dict[str, Dict[str, Any]] = {}

def is_safe_non_neg_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool) and 0 <= val <= (2**53 - 1)

def is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val)

def compact_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

@app.post("/quantize")
async def quantize_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    phase = body.get("phase")
    if phase == "freeze":
        return handle_freeze(body)
    elif phase == "select":
        return handle_select(body)
    else:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

def handle_freeze(body: Dict[str, Any]) -> JSONResponse:
    freeze_id = body.get("freezeId")
    calib_dig = body.get("calibrationDigest")
    tok_dig = body.get("tokenizerDigest")
    allowed_reasons = body.get("allowedUnsupportedReasons")
    candidates = body.get("candidates")

    # Validation of required request fields
    if not isinstance(freeze_id, str) or not (1 <= len(freeze_id) <= 128):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(calib_dig, str) or len(calib_dig) == 0 or not isinstance(tok_dig, str) or len(tok_dig) == 0:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(allowed_reasons, list) or not all(isinstance(r, str) and len(r) > 0 for r in allowed_reasons):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
    if len(allowed_reasons) != len(set(allowed_reasons)):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(candidates, list) or len(candidates) == 0:
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    # Check Replay & Conflict
    if freeze_id in STORE:
        stored_entry = STORE[freeze_id]
        if stored_entry["request"] == body:
            return JSONResponse(status_code=200, content=stored_entry["response"])
        else:
            return JSONResponse(status_code=409, content={"error": "FREEZE_ID_CONFLICT"})

    allowed_reasons_set = set(allowed_reasons)
    cand_names_seen = set()
    processed_candidates = []

    for c in candidates:
        reason_codes = set()
        if not isinstance(c, dict):
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

        cname = c.get("name")
        cfiles = c.get("files")
        cloadable = c.get("loadable")
        c_calib = c.get("calibrationDigest")
        c_tok = c.get("tokenizerDigest")
        c_unsupported = c.get("unsupportedReason")

        if not isinstance(cname, str) or len(cname) == 0 or cname in cand_names_seen:
            return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})
        cand_names_seen.add(cname)

        inventory = []
        total_bytes = None
        pkg_digest = None

        # Files inventory calculation
        if not isinstance(cfiles, dict) or len(cfiles) == 0:
            reason_codes.add("INVALID_INPUT")
        else:
            files_valid = True
            for fname, fcontent in cfiles.items():
                if not isinstance(fname, str) or not isinstance(fcontent, str):
                    files_valid = False
                    break
                fbytes = len(fcontent.encode("utf-8"))
                fsha = hashlib.sha256(fcontent.encode("utf-8")).hexdigest().lower()
                inventory.append({"name": fname, "bytes": fbytes, "sha256": fsha})

            if not files_valid:
                reason_codes.add("INVALID_INPUT")
                inventory = []
            else:
                inventory.sort(key=lambda x: x["name"].encode("utf-8"))
                total_bytes = sum(item["bytes"] for item in inventory)
                pkg_digest = hashlib.sha256(compact_json_bytes(inventory)).hexdigest()

        # Status & Code determination
        status = "invalid"
        if isinstance(c_unsupported, str) and len(c_unsupported) > 0:
            if c_unsupported in allowed_reasons_set:
                status = "unsupported"
            else:
                status = "invalid"
                reason_codes.add("UNALLOWED_UNSUPPORTED_REASON")
        else:
            if cloadable is not True:
                reason_codes.add("NOT_LOADABLE")
            if c_calib != calib_dig:
                reason_codes.add("CALIBRATION_MISMATCH")
            if c_tok != tok_dig:
                reason_codes.add("TOKENIZER_MISMATCH")

            if len(reason_codes) == 0:
                status = "frozen"
            else:
                status = "invalid"

        if status == "invalid" and "INVALID_INPUT" in reason_codes and not isinstance(cfiles, dict):
            inventory = []
            total_bytes = None
            pkg_digest = None

        sorted_codes = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))

        processed_candidates.append({
            "name": cname,
            "status": status,
            "inventory": inventory,
            "totalBytes": total_bytes,
            "packageDigest": pkg_digest,
            "reasonCodes": sorted_codes
        })

    # Candidates sorted by name UTF-8 bytes
    processed_candidates.sort(key=lambda x: x["name"].encode("utf-8"))

    response_payload = {
        "freezeId": freeze_id,
        "candidates": processed_candidates
    }

    STORE[freeze_id] = {
        "request": body,
        "response": response_payload
    }

    return JSONResponse(status_code=200, content=response_payload)

def handle_select(body: Dict[str, Any]) -> JSONResponse:
    freeze_id = body.get("freezeId")
    candidates = body.get("candidates")
    policy = body.get("policy")
    latencies = body.get("latencies")
    rows = body.get("rows")

    # Missing candidates, rows, or policy returns HTTP 400
    if not isinstance(candidates, list) or not isinstance(rows, list) or not isinstance(policy, dict):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    if not isinstance(freeze_id, str) or not (1 <= len(freeze_id) <= 128):
        return JSONResponse(status_code=400, content={"error": "INVALID_INPUT"})

    max_bytes = policy.get("maxBytes")
    agg_floor = policy.get("aggregateFloor")
    req_slices = policy.get("requiredSlices")
    max_lat_ms = policy.get("maxLatencyMs")
    cand_order = policy.get("candidateOrder")

    policy_valid = (
        is_safe_non_neg_int(max_bytes) and
        is_finite_number(agg_floor) and 0.0 <= float(agg_floor) <= 1.0 and
        isinstance(req_slices, dict) and
        is_finite_number(max_lat_ms) and float(max_lat_ms) >= 0.0 and
        isinstance(cand_order, list) and len(cand_order) > 0 and
        all(isinstance(c, str) and len(c) > 0 for c in cand_order) and
        len(cand_order) == len(set(cand_order))
    )

    if policy_valid and isinstance(req_slices, dict):
        for sname, sfloor in req_slices.items():
            if not (isinstance(sname, str) and is_finite_number(sfloor) and 0.0 <= float(sfloor) <= 1.0):
                policy_valid = False
                break

    # Lineage verification
    lineage_valid = True
    stored_candidates = None

    if freeze_id not in STORE:
        lineage_valid = False
    else:
        stored_resp = STORE[freeze_id]["response"]
        stored_candidates = stored_resp.get("candidates")
        if stored_candidates != candidates:
            lineage_valid = False

    # Collect candidate names from supplied candidates
    cand_map: Dict[str, dict] = {}
    if isinstance(candidates, list):
        for c in candidates:
            if isinstance(c, dict) and isinstance(c.get("name"), str):
                cand_map[c["name"]] = c

    if policy_valid:
        if set(cand_order) != set(cand_map.keys()):
            policy_valid = False

    results = []
    admitted_candidates = []

    order_to_use = cand_order if policy_valid else sorted(list(cand_map.keys()), key=lambda x: x.encode("utf-8"))

    for cname in order_to_use:
        reason_codes = set()
        c_obj = cand_map.get(cname)

        if freeze_id not in STORE:
            reason_codes.add("NOT_FROZEN")
        elif not lineage_valid:
            reason_codes.add("INVALID_LINEAGE")

        if not policy_valid:
            reason_codes.add("INVALID_POLICY")

        # Manifest & Inventory recomputation
        total_bytes = None
        pkg_digest = None
        c_status = None

        if c_obj and isinstance(c_obj.get("inventory"), list):
            inv = c_obj["inventory"]
            c_status = c_obj.get("status")
            if len(inv) > 0:
                total_bytes = sum(item.get("bytes", 0) for item in inv)
                pkg_digest = hashlib.sha256(compact_json_bytes(inv)).hexdigest()

        if total_bytes is None or pkg_digest is None or c_obj.get("packageDigest") != pkg_digest:
            reason_codes.add("INVALID_MANIFEST")

        # Latency lookup
        c_latency = None
        if isinstance(latencies, dict):
            lat_val = latencies.get(cname)
            if is_finite_number(lat_val) and float(lat_val) >= 0.0:
                c_latency = float(lat_val)

        if c_latency is None:
            reason_codes.add("INVALID_POLICY")

        # Predictions & Accuracies
        preds_valid = True
        candidate_preds = []

        if not isinstance(rows, list) or len(rows) == 0:
            preds_valid = False
        else:
            for r in rows:
                if not isinstance(r, dict):
                    preds_valid = False
                    break
                r_label = r.get("label")
                r_slice = r.get("slice")
                r_preds = r.get("predictions")

                if r_label not in (0, 1) or not isinstance(r_slice, str) or len(r_slice) == 0 or not isinstance(r_preds, dict):
                    preds_valid = False
                    break

                pred_val = r_preds.get(cname)
                if pred_val not in (0, 1):
                    preds_valid = False
                    break

                candidate_preds.append({
                    "label": int(r_label),
                    "slice": r_slice,
                    "pred": int(pred_val)
                })

        agg_acc = None
        slice_accs = None

        if not preds_valid:
            reason_codes.add("INVALID_PREDICTIONS")
            agg_acc = None
            slice_accs = None
        else:
            total_rows = len(candidate_preds)
            correct_count = sum(1 for cp in candidate_preds if cp["label"] == cp["pred"])
            agg_acc = round(correct_count / total_rows, 12)

            if policy_valid and agg_acc < float(agg_floor):
                reason_codes.add("AGGREGATE_FLOOR")

            slice_accs = {}
            if policy_valid and isinstance(req_slices, dict):
                slice_groups: Dict[str, List[dict]] = {}
                for cp in candidate_preds:
                    slice_groups.setdefault(cp["slice"], []).append(cp)

                for req_name, req_floor in req_slices.items():
                    if req_name not in slice_groups:
                        reason_codes.add(f"MISSING_SLICE:{req_name}")
                    else:
                        s_rows = slice_groups[req_name]
                        s_correct = sum(1 for cp in s_rows if cp["label"] == cp["pred"])
                        s_acc = round(s_correct / len(s_rows), 12)
                        slice_accs[req_name] = s_acc
                        if s_acc < float(req_floor):
                            reason_codes.add(f"SLICE_FLOOR:{req_name}")

        # Limits Check
        if policy_valid and total_bytes is not None and total_bytes > max_bytes:
            reason_codes.add("SIZE_LIMIT")

        if policy_valid and c_latency is not None and c_latency > float(max_lat_ms):
            reason_codes.add("LATENCY_LIMIT")

        is_admitted = (c_status == "frozen") and (len(reason_codes) == 0)

        sorted_codes = sorted(list(reason_codes), key=lambda x: x.encode("utf-8"))

        res_item = {
            "name": cname,
            "aggregate": agg_acc,
            "slices": slice_accs,
            "totalBytes": total_bytes,
            "latencyMs": c_latency,
            "admitted": is_admitted,
            "reasonCodes": sorted_codes,
            "_order_idx": cand_order.index(cname) if policy_valid and cname in cand_order else 999,
            "_raw_candidate_obj": c_obj
        }

        results.append(res_item)
        if is_admitted:
            admitted_candidates.append(res_item)

    selected_name = None
    package_manifest = None

    if admitted_candidates:
        # Tie-breaker: smaller bytes -> lower latency -> candidateOrder index
        admitted_candidates.sort(key=lambda x: (
            x["totalBytes"] if x["totalBytes"] is not None else float("inf"),
            x["latencyMs"] if x["latencyMs"] is not None else float("inf"),
            x["_order_idx"]
        ))
        winner = admitted_candidates[0]
        selected_name = winner["name"]
        package_manifest = winner["_raw_candidate_obj"]

    # Format final results list (stripping internal sorting keys)
    final_results = []
    for r in results:
        final_results.append({
            "name": r["name"],
            "aggregate": r["aggregate"],
            "slices": r["slices"],
            "totalBytes": r["totalBytes"],
            "latencyMs": r["latencyMs"],
            "admitted": r["admitted"],
            "reasonCodes": r["reasonCodes"]
        })

    return JSONResponse(status_code=200, content={
        "freezeId": freeze_id,
        "selected": selected_name,
        "results": final_results,
        "packageManifest": package_manifest
    })

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Quantized Model Candidate Admission"}
