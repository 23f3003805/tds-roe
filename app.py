"""
Korean Speech Dataset API
=========================
Receives audio (base64) → identifies the dataset row → returns dataset statistics as JSON.

DEPLOYMENT: Run on Render / Railway / any cloud platform.
            Or use ngrok locally for testing.
"""

import base64
import hashlib
import json
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from scipy import stats as scipy_stats

app = Flask(__name__)

# ─────────────────────────────────────────────
# STEP 1: Load the dataset ONCE at startup
# ─────────────────────────────────────────────
# We use the Mozilla Common Voice Korean dataset (version 11).
# Change DATASET_NAME / DATASET_CONFIG if your assignment uses a different one.

DATASET_NAME   = "mozilla-foundation/common_voice_11_0"
DATASET_CONFIG = "ko"
DATASET_SPLIT  = "train"

print(f"Loading dataset {DATASET_NAME} ({DATASET_CONFIG}) ...")
try:
    from datasets import load_dataset
    raw_ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT)
    df_full = raw_ds.to_pandas()
    # Drop the raw audio column for stats (we keep it separately for matching)
    AUDIO_COL = "audio"   # HuggingFace stores audio as dict with 'array' + 'path' + 'sampling_rate'
    print(f"Dataset loaded: {len(df_full)} rows, columns: {list(df_full.columns)}")
except Exception as e:
    print(f"WARNING: Could not load dataset at startup: {e}")
    df_full = pd.DataFrame()   # empty fallback so server still starts

# ─────────────────────────────────────────────
# STEP 2: Build an index  audio_bytes_hash → row_index
#         so we can look up a row from incoming audio quickly.
# ─────────────────────────────────────────────

audio_hash_to_idx: dict[str, int] = {}

def _hash_audio_array(arr) -> str:
    """Hash a numpy audio array to a short hex string for fast lookup."""
    return hashlib.md5(np.array(arr).tobytes()).hexdigest()

if not df_full.empty and AUDIO_COL in df_full.columns:
    print("Building audio hash index …")
    for i, row in df_full.iterrows():
        try:
            h = _hash_audio_array(row[AUDIO_COL]["array"])
            audio_hash_to_idx[h] = i
        except Exception:
            pass
    print(f"Index built: {len(audio_hash_to_idx)} entries")


# ─────────────────────────────────────────────
# STEP 3: Statistics helpers
# ─────────────────────────────────────────────

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _safe_mode(series: pd.Series):
    """Return mode value (first if tie), or None if empty."""
    m = series.mode()
    return m.iloc[0] if len(m) > 0 else None

def compute_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Compute the required statistics over all columns of the DataFrame.
    Returns a dict matching the required JSON structure.
    """
    # Work on a copy without the raw audio column (not meaningful for stats)
    work_df = df.drop(columns=[AUDIO_COL], errors="ignore")

    numeric_cols   = [c for c in work_df.columns if _is_numeric(work_df[c])]
    categoric_cols = [c for c in work_df.columns if not _is_numeric(work_df[c])]

    rows    = len(work_df)
    columns = list(work_df.columns)

    # ── Per-column stats (only for numeric cols) ──
    mean      = {}
    std       = {}
    variance  = {}
    col_min   = {}
    col_max   = {}
    median    = {}
    mode      = {}
    col_range = {}

    for c in numeric_cols:
        s = work_df[c].dropna()
        mean[c]      = float(s.mean())
        std[c]       = float(s.std())
        variance[c]  = float(s.var())
        col_min[c]   = float(s.min())
        col_max[c]   = float(s.max())
        median[c]    = float(s.median())
        mode[c]      = float(_safe_mode(s)) if _safe_mode(s) is not None else None
        col_range[c] = float(s.max() - s.min())

    # ── allowed_values: unique values for CATEGORICAL columns ──
    allowed_values = {}
    for c in categoric_cols:
        uniq = work_df[c].dropna().unique().tolist()
        # Convert numpy types to plain Python
        allowed_values[c] = [str(v) for v in sorted(uniq)]

    # ── value_range: [min, max] for NUMERIC columns ──
    value_range = {}
    for c in numeric_cols:
        s = work_df[c].dropna()
        value_range[c] = [float(s.min()), float(s.max())]

    # ── correlation matrix (numeric cols only) ──
    if len(numeric_cols) >= 2:
        corr_matrix = work_df[numeric_cols].corr().round(6)
        correlation = corr_matrix.values.tolist()   # list of lists
    else:
        correlation = []

    return {
        "rows":           rows,
        "columns":        columns,
        "mean":           mean,
        "std":            std,
        "variance":       variance,
        "min":            col_min,
        "max":            col_max,
        "median":         median,
        "mode":           mode,
        "range":          col_range,
        "allowed_values": allowed_values,
        "value_range":    value_range,
        "correlation":    correlation,
    }


# Pre-compute stats for the full dataset so we only do it once
_cached_stats = None

def get_stats() -> dict:
    global _cached_stats
    if _cached_stats is None and not df_full.empty:
        print("Computing dataset statistics (one-time) …")
        _cached_stats = compute_dataset_stats(df_full)
        print("Stats ready.")
    return _cached_stats or {}


# ─────────────────────────────────────────────
# STEP 4: The API endpoint
# ─────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return jsonify({"status": "ok", "message": "Send POST with audio_base64"})
    """
    Receives:  {"audio_id": "q0", "audio_base64": "<base64 string>"}
    Returns:   The dataset statistics JSON
    """
    body = request.get_json(force=True)

    audio_id     = body.get("audio_id", "")
    audio_b64    = body.get("audio_base64", "")

    # Decode the incoming audio
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return jsonify({"error": "invalid base64"}), 400

    # (Optional) verify we can find this audio in our index
    # If the dataset is loaded, try to match the audio to a specific row.
    # For this assignment the stats are for the WHOLE dataset (same for every audio),
    # so matching is only needed if stats differ per-row.
    # If your assignment requires per-row stats, uncomment the block below.
    #
    # import soundfile as sf, io
    # audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    # h = _hash_audio_array(audio_array)
    # row_idx = audio_hash_to_idx.get(h)
    # if row_idx is None:
    #     return jsonify({"error": "audio not found in dataset"}), 404

    stats = get_stats()
    if not stats:
        return jsonify({"error": "dataset not loaded"}), 500

    return jsonify(stats)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "rows_loaded": len(df_full)})


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
