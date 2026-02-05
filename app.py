from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
import uvicorn
import traceback
import base64
from typing import Optional

# ...existing imports...
from final_inference_logic import predict, load_models, generate_layman_explanation
from schemas import PredictRequest, PredictResponse, SUPPORTED_LANGS

app = FastAPI(title="Voice Detector API")

# ...existing startup event...
@app.on_event("startup")
def load_models_on_startup():
    # allow skipping heavy model load in test/dev via env var
    if os.environ.get("SKIP_MODEL_LOAD", "0") == "1":
        app.state.stage1 = None
        app.state.stage2 = None
        app.state.models_ready = False
        return
    try:
        app.state.stage1, app.state.stage2 = load_models()
        app.state.models_ready = True
    except Exception:
        # don't crash the process on model load failure; mark not ready
        app.state.stage1 = None
        app.state.stage2 = None
        app.state.models_ready = False


# ...existing /health endpoint...


# existing multipart /predict remains but ensure response_model when possible
@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(...), language: Optional[str] = None):
    if not app.state.models_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        label, score, reason, diag = predict(tmp.name, stage1=app.state.stage1, stage2=app.state.stage2)
        layman = generate_layman_explanation(label, diag)
        resp = PredictResponse(label=label, confidence=float(score), explanation=reason, layman_explanation=layman, language=(language.lower() if language else None))
        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        if tmp is not None and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


@app.post("/predict_base64", response_model=PredictResponse)
async def predict_base64(req: PredictRequest):
    """Accept JSON with base64-encoded audio and optional language."""
    if not app.state.models_ready:
        raise HTTPException(status_code=503, detail="Models not loaded")

    b64 = req.audio_base64.strip()
    # strip data URL prefix if present
    if b64.startswith("data:"):
        try:
            b64 = b64.split(",", 1)[1]
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid data URL")

    try:
        audio_bytes = base64.b64decode(b64)
    except Exception:
        raise HTTPException(status_code=400, detail="audio_base64 is not valid base64")

    # determine suffix from filename if given, fall back to .mp3
    suffix = (os.path.splitext(req.filename)[1] if req.filename else "") or ".mp3"
    tmp = None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()

        label, score, reason, diag = predict(tmp.name, stage1=app.state.stage1, stage2=app.state.stage2)
        layman = generate_layman_explanation(label, diag)

        resp = PredictResponse(
            label=label,
            confidence=float(score),
            explanation=reason,
            layman_explanation=layman,
            language=(req.language.lower() if req.language else None)
        )
        return resp

    finally:
        if tmp is not None and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except Exception:
                pass


# ...existing if __name__ == "__main__": ...
