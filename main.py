from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from hashlib import sha256
import motor.motor_asyncio

from ocr_validator import run_text_validation  

from  image_classifier import predict_from_base64
from fastapi import APIRouter
from pydantic import BaseModel
from image_classifier import predict_from_base64
from ocr_validator import run_text_validation


app = FastAPI(
    title="College ID Validator API",
    description="API for validating college ID cards using OCR and AI",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head><title>College ID Validator API</title></head>
        <body>
            <h1>Welcome to the College ID Validator API</h1>
            <p>Use the <code>/validate-id</code> POST endpoint to validate ID cards.</p>
            <p>Go to <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """





router = APIRouter()

class IDValidationRequest(BaseModel):
    image_base64: str
    user_id: str

@app.post("/validate-id")
async def validate_id(data: IDValidationRequest):
    genuine_confidence = 0.0
    ocr_confidence = 0.0
    face_photo_found = False
    threshold = 0.70

    try:
        result = predict_from_base64(data.image_base64)
        genuine_confidence = float(result.get("genuine_confidence", 0.0))
        predicted_class = result.get("final_label", "non-id")

        
        if predicted_class == "non-id":
            return {
                "user_id": data.user_id,
                "validation_score": 0.0,
                "label": "fake",
                "status": "rejected",
                "reason": "Uploaded image is not recognized as a valid ID card",
                "threshold": threshold
            }

    except Exception as e_img:
        print("Image prediction error:", e_img)

    try:
        ocr_results = run_text_validation(data.image_base64, data.user_id)
        ocr_confidence = float(ocr_results.get("ocr_confidence", 0.0))
        face_photo_found = ocr_results.get("validation", {}).get("face_photo_found", False)
    except Exception as e_ocr:
        print("OCR error:", e_ocr)

    validation_score = round(0.7 * genuine_confidence + 0.3 * ocr_confidence, 4)

    if ocr_confidence == 0.0 or validation_score < 0.6:
        label = "fake"
        status = "rejected"
        reason = "Template mismatch and low OCR confidence"
    elif validation_score > 0.8:
        label = "genuine"
        status = "approved"
        reason = "High confidence from image and OCR validation"
    else:
        label = "suspicious"
        status = "manual_review"
        reason = "Moderate score or low OCR confidence"

    if not face_photo_found and label != "fake":
        label = "suspicious"
        status = "manual_review"
        reason = "Face photo not confidently detected — needs review"

    return {
        "user_id": data.user_id,
        "validation_score": validation_score,
        "label": label,
        "status": status,
        "reason": reason,
        "threshold": threshold
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/version")
async def version_info():
    return {
        "version": "1.0.0",
        "model": "AI ID Validator",
        "last_updated": "2025-05-20"
    }
