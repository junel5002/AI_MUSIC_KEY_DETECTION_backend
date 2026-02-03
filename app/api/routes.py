from fastapi import APIRouter, UploadFile, File
import tempfile
import shutil

from app.services.audio_loader import load_audio
from app.services.feature_extractor import extract_chroma
from app.services.rule_based import detect_key

router = APIRouter(prefix="/analyze", tags=["Analysis"])


# =========================
# UPLOAD AUDIO (used by Upload page)
# =========================
@router.post("/file")
async def analyze_file(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Load & analyze
    y, sr = load_audio(tmp_path)
    chroma = extract_chroma(y, sr)
    key, confidence = detect_key(chroma)

    return {
        "key": key,
        "confidence": confidence,
        "method": "upload-file"
    }


# =========================
# EXISTING UPLOAD ENDPOINT (kept, not removed)
# =========================
@router.post("/upload")
async def analyze_audio(file: UploadFile = File(...)):
    y, sr = load_audio(file)
    chroma = extract_chroma(y, sr)
    key, confidence = detect_key(chroma)

    return {
        "key": key,
        "confidence": confidence,
        "method": "rule-based"
    }


# =========================
# LIVE MICROPHONE AUDIO
# =========================
@router.post("/live")
async def analyze_live_audio(file: UploadFile = File(...)):
    y, sr = load_audio(file)
    chroma = extract_chroma(y, sr)
    key, confidence = detect_key(chroma)

    return {
        "key": key,
        "confidence": confidence,
        "method": "live-mic"
    }
