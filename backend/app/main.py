from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env", override=True)
MODEL_DIR = Path(os.getenv("CHESTXRAY_MODEL_DIR", str(ROOT / "models"))).resolve()
ARTIFACT_DIR = Path(os.getenv("CHESTXRAY_ARTIFACT_DIR", str(BACKEND / "artifacts"))).resolve()
DATA_DIR = Path(os.getenv("CHESTXRAY_DATA_DIR", str(BACKEND / "data"))).resolve()
DB_PATH = DATA_DIR / "chestxray_lab.db"
MAX_UPLOAD_BYTES = int(os.getenv("CHESTXRAY_MAX_UPLOAD_MB", "12")) * 1024 * 1024
LABELS = ["NORMAL", "PNEUMONIA"]

app = FastAPI(title="X-Doc", version="1.0.0", description="Local educational CNN experimentation tool. Predictions are not medical diagnoses.")
origins = [o.strip() for o in os.getenv("CHESTXRAY_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


class PredictionResult(BaseModel):
    model_key: str
    model_name: str
    predicted_label: str
    scores: dict[str, float]
    inference_ms: float
    input_size: str = "224x224 RGB, pixel values scaled to [0, 1]"


class Health(BaseModel):
    status: str
    database: str


_models: dict[str, Any] = {}
_load_errors: dict[str, str] = {}
_caffe_layer: Any = None


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


@contextmanager
def database():
    db = connect()
    try:
        yield db
        db.commit()
    finally:
        db.close()


def init_db() -> None:
    with database() as db:
        db.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL, model_key TEXT NOT NULL,
            original_filename TEXT, predicted_label TEXT NOT NULL, scores_json TEXT NOT NULL,
            inference_ms REAL NOT NULL, input_width INTEGER, input_height INTEGER, status TEXT NOT NULL,
            error_message TEXT)""")


@app.on_event("startup")
def startup() -> None:
    init_db()


def discover_model_paths() -> dict[str, Path]:
    if not MODEL_DIR.is_dir():
        return {}
    names = {
        "grid_021.keras": ("custom_cnn", "Custom CNN (grid_021)"),
        "VGG16_frozen.keras": ("vgg16_frozen", "VGG16 - frozen backbone"),
        "VGG16_fine_tuned.keras": ("vgg16_fine_tuned", "VGG16 - fine-tuned"),
        "ResNet50_frozen.keras": ("resnet50_frozen", "ResNet50 - frozen backbone"),
        "ResNet50_fine_tuned.keras": ("resnet50_fine_tuned", "ResNet50 - fine-tuned"),
    }
    found = {}
    paths = {path.name: path for path in MODEL_DIR.glob("*.keras")}
    preferred = ["VGG16_fine_tuned.keras", "VGG16_frozen.keras", "ResNet50_fine_tuned.keras", "ResNet50_frozen.keras", "grid_021.keras"]
    for filename in preferred + sorted(name for name in paths if name not in preferred):
        path = paths.get(filename)
        if not path:
            continue
        key = names.get(path.name, (path.stem.lower(), path.stem.replace("_", " ").title()))[0]
        found[key] = path
    return found


def model_catalog() -> list[dict[str, Any]]:
    catalog = []
    comparison = {}
    metrics_path = ARTIFACT_DIR / "model_comparison.csv"
    if metrics_path.exists():
        try:
            with metrics_path.open(newline="", encoding="utf-8-sig") as stream:
                for row in csv.DictReader(stream):
                    comparison[row.get("Model", row.get("model", "")).strip().lower()] = row
        except (OSError, csv.Error):
            pass
    for key, path in discover_model_paths().items():
        metric = comparison.get("custom_cnn" if key == "custom_cnn" else path.stem.lower(), {})
        catalog.append({"key": key, "name": _display_name(path), "file": path.name,
                        "available": key not in _load_errors, "status": "Detected; loads on demand" if key not in _load_errors else "Unavailable",
                        "error": _load_errors.get(key), "metrics": {
                            "accuracy": metric.get("Accuracy"), "roc_auc": metric.get("ROC-AUC"),
                            "macro_f1": metric.get("Macro F1")}})
    return catalog


def _display_name(path: Path) -> str:
    names = {"grid_021.keras": "Custom CNN", "VGG16_frozen.keras": "VGG16 - frozen backbone",
             "VGG16_fine_tuned.keras": "VGG16 - fine-tuned", "ResNet50_frozen.keras": "ResNet50 - frozen backbone",
             "ResNet50_fine_tuned.keras": "ResNet50 - fine-tuned"}
    return names.get(path.name, path.stem.replace("_", " ").title())


@app.get("/api/health", response_model=Health)
def health() -> Health:
    try:
        with database() as db:
            db.execute("SELECT 1")
        return Health(status="ok", database="ok")
    except sqlite3.Error:
        raise HTTPException(503, detail="Local prediction database is unavailable.")


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": model_catalog(), "model_directory": str(MODEL_DIR), "count": len(discover_model_paths())}


def get_model(key: str) -> Any:
    paths = discover_model_paths()
    if key not in paths:
        raise HTTPException(404, detail=f"Model '{key}' was not found in the configured model folder.")
    if key not in _models and key not in _load_errors:
        try:
            import tensorflow as tf
            custom_objects = _keras_custom_objects(tf)
            model = tf.keras.models.load_model(paths[key], compile=False, safe_mode=True,
                                               custom_objects=custom_objects)
            shape = model.input_shape
            if isinstance(shape, list):
                raise ValueError("Multiple model inputs are not supported.")
            if len(shape) != 4 or tuple(shape[1:]) != (224, 224, 3):
                raise ValueError(f"Expected input shape (None, 224, 224, 3), found {shape}.")
            _models[key] = model
        except Exception as exc:
            _load_errors[key] = str(exc)
    if key in _load_errors:
        raise HTTPException(503, detail=f"{_display_name(paths[key])} could not be loaded. Check TensorFlow compatibility and model file: {_load_errors[key][:240]}")
    return _models[key]


def _keras_custom_objects(tf: Any) -> dict[str, Any]:
    global _caffe_layer
    if _caffe_layer is None:
        @tf.keras.utils.register_keras_serializable(package="ChestXray")
        class CaffePreprocessing(tf.keras.layers.Layer):
            def call(self, inputs: Any) -> Any:
                pixels = inputs * 255.0
                bgr = tf.reverse(pixels, axis=[-1])
                means = tf.constant([103.939, 116.779, 123.68], dtype=inputs.dtype)
                return bgr - means
        _caffe_layer = CaffePreprocessing
    return {
        "CaffePreprocessing": _caffe_layer,
        "ChestXray>CaffePreprocessing": _caffe_layer,
    }


@app.post("/api/predict")
async def predict(image: UploadFile = File(...), model_keys: str = Form(...)) -> dict[str, Any]:
    filename = Path(image.filename or "image").name[:200]
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(415, detail="Use a PNG, JPEG, or WebP image.")
            if source.width * source.height > 60_000_000:
                raise HTTPException(413, detail="Image dimensions are too large to process safely.")
            source.load()
            width, height = source.size
            arr = np.asarray(source.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(400, detail="The uploaded file is not a valid supported image.")
    keys = list(dict.fromkeys(k.strip() for k in model_keys.split(",") if k.strip()))
    if not keys:
        raise HTTPException(422, detail="Select at least one available model.")
    results = []
    for key in keys:
        model = get_model(key)
        start = time.perf_counter()
        try:
            output = np.asarray(model.predict(arr[None, ...], verbose=0)).reshape(-1)
            if output.size != 1 or not np.isfinite(output[0]):
                raise ValueError("Expected one finite sigmoid score from the binary classifier.")
            score = float(np.clip(output[0], 0, 1))
        except Exception as exc:
            raise HTTPException(500, detail=f"Inference failed for {key}: {str(exc)[:200]}")
        elapsed = (time.perf_counter() - start) * 1000
        result = PredictionResult(model_key=key, model_name=next(x["name"] for x in model_catalog() if x["key"] == key),
                                  predicted_label=LABELS[int(score >= 0.5)], scores={"NORMAL": 1.0-score, "PNEUMONIA": score}, inference_ms=round(elapsed, 2))
        results.append(result.model_dump())
        with database() as db:
            db.execute("INSERT INTO predictions (created_at, model_key, original_filename, predicted_label, scores_json, inference_ms, input_width, input_height, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (datetime.now(timezone.utc).isoformat(), key, filename, result.predicted_label, json.dumps(result.scores), result.inference_ms, width, height, "success"))
    return {"results": results, "notice": "Model predictions are experimental scores and are not medical diagnoses or clinically validated probabilities."}


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    path = ARTIFACT_DIR / "model_comparison.csv"
    if not path.exists():
        return {"available": False, "split": None, "rows": [], "message": "No exported evaluation metrics found. Add model_comparison.csv to backend/artifacts/."}
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if rows and not any(k.lower() in {"model", "model_key"} for k in rows[0]):
            raise HTTPException(422, detail="model_comparison.csv needs a Model or model_key column.")
        return {"available": bool(rows), "split": "test" if rows else None, "rows": rows, "message": None if rows else "The metrics file contains no rows."}
    except (OSError, csv.Error):
        raise HTTPException(422, detail="Could not read model_comparison.csv. Check the CSV format.")


@app.get("/api/metrics/{model_key}")
def model_metrics(model_key: str) -> dict[str, Any]:
    data = metrics()
    rows = [r for r in data["rows"] if r.get("model", "").lower() == model_key.lower() or r.get("model_key", "").lower() == model_key.lower()]
    return {"available": bool(rows), "rows": rows, "split": data["split"]}


@app.get("/api/errors")
def errors(error_type: str | None = Query(None, pattern="^(false_positive|false_negative)$"), model: str | None = None) -> dict[str, Any]:
    path = ARTIFACT_DIR / "test_predictions.csv"
    if not path.exists():
        # Notebook exports one predictions file per model; read only prediction/label columns, never expose local source paths.
        files = list(ARTIFACT_DIR.glob("*_predictions.csv")) if ARTIFACT_DIR.exists() else []
        records = []
        for file in files:
            key = file.stem.removesuffix("_predictions").lower()
            if model and model.lower() not in key: continue
            try:
                with file.open(newline="", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    required = {"actual_label", "predicted_label"}
                    if not required.issubset(set(reader.fieldnames or [])):
                        raise HTTPException(422, detail=f"{file.name} needs actual_label and predicted_label columns.")
                    for row in reader:
                        actual = _label(row.get("actual_label", ""))
                        predicted = _label(row.get("predicted_label", ""))
                        if actual == predicted: continue
                        typ = "false_positive" if actual == "NORMAL" else "false_negative"
                        if error_type and error_type != typ: continue
                        score = row.get("pneumonia_score")
                        records.append({"model": key, "actual": actual, "predicted": predicted, "score": score})
            except (OSError, csv.Error): continue
        return {"available": bool(records), "rows": records, "message": None if records else "No exported test prediction records found. Add per-model *_predictions.csv files to backend/artifacts/."}
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if not ({"actual_label", "actual"} & fields) or not ({"predicted_label", "predicted"} & fields):
                raise HTTPException(422, detail="test_predictions.csv needs actual/predicted label columns.")
            rows = list(reader)
        output = []
        for row in rows:
            actual = _label(row.get("actual_label", row.get("actual", ""))); predicted = _label(row.get("predicted_label", row.get("predicted", "")))
            typ = "false_positive" if actual == "NORMAL" and predicted == "PNEUMONIA" else "false_negative"
            if actual == predicted or error_type and error_type != typ or model and model.lower() not in row.get("model", row.get("model_key", "")).lower(): continue
            output.append({"model": row.get("model", row.get("model_key", "unknown")), "actual": actual, "predicted": predicted, "score": row.get("pneumonia_score", row.get("score"))})
        return {"available": bool(output), "rows": output, "message": "No misclassified records match these filters." if not output else None}
    except (OSError, csv.Error):
        raise HTTPException(422, detail="Could not read test_predictions.csv.")


def _label(value: str) -> str:
    cleaned = str(value).strip().upper()
    if cleaned in {"0", "0.0", "NORMAL"}: return "NORMAL"
    if cleaned in {"1", "1.0", "PNEUMONIA"}: return "PNEUMONIA"
    return cleaned


@app.get("/api/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    path = ARTIFACT_DIR / "dataset_stats.json"
    if not path.exists():
        return {"available": False, "stats": None, "message": "No exported dataset statistics found. Add dataset_stats.json to backend/artifacts/."}
    try:
        return {"available": True, "stats": json.loads(path.read_text(encoding="utf-8")), "message": None}
    except (OSError, json.JSONDecodeError):
        raise HTTPException(422, detail="Could not read dataset_stats.json. Check its JSON format.")


@app.get("/api/predictions")
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict[str, Any]:
    with database() as db:
        rows = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    return {"total": total, "rows": [{**dict(r), "scores": json.loads(r["scores_json"])} for r in rows]}


@app.get("/api/predictions/{prediction_id}")
def prediction_detail(prediction_id: int) -> dict[str, Any]:
    with database() as db:
        row = db.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,)).fetchone()
    if row is None: raise HTTPException(404, detail="Prediction record not found.")
    record = dict(row); record["scores"] = json.loads(record.pop("scores_json"))
    return record


@app.delete("/api/predictions")
def clear_history() -> dict[str, Any]:
    with database() as db: db.execute("DELETE FROM predictions")
    return {"deleted": True}
