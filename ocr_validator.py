from PIL import Image
import pytesseract
import base64
import io
import re
import cv2
import numpy as np
from rapidfuzz import fuzz
import platform

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"


KNOWN_COLLEGES = [
    "JNTU Hyderabad",
    "NIT Warangal",
    "IIT Bombay"
]



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def correct_orientation(image: Image.Image, save_path="corrected_image.jpg"):
    try:
        osd = pytesseract.image_to_osd(image)
        rotation = int(re.search(r'Rotate: (\d+)', osd).group(1))
        if rotation != 0:
            image = image.rotate(-rotation, expand=True)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(save_path)
    except Exception as e:
        print(f"Rotation detection failed: {e}")
    return image

def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return correct_orientation(image)
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def extract_text_with_confidence(image):
    try:
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        texts = ocr_data['text']
        confs = ocr_data['conf']
        valid_confs = [int(conf) for conf, txt in zip(confs, texts) if txt.strip() and conf != '-1']
        avg_conf = sum(valid_confs) / len(valid_confs) if valid_confs else 0
        full_text = " ".join([txt for txt in texts if txt.strip()])
        return full_text, avg_conf
    except Exception as e:
        print(f"OCR error: {e}")
        return "", 0

def preprocess_text(text):
    return ' '.join(text.split()).lower()

def validate_college_name(text, threshold=75):
    text = text.lower().strip()
    best_score = 0
    best_match = None

    for college in KNOWN_COLLEGES:
        score = fuzz.token_set_ratio(college.lower(), text)

        if text in college.lower():
            score += 10

        if score > best_score:
            best_score = score
            best_match = college


    return (True, best_match) if best_score >= threshold else (False, None)


def validate_mandatory_fields(text):
    roll_match = re.search(r"roll\s*no\.?\s*[:\-]?\s*([\w\d\-]+)", text, re.IGNORECASE)
    year_match = re.search(r"(first|second|third|fourth)\s+year", text, re.IGNORECASE)
    course_match = re.search(r"\b(btech|b\.tech|mtech|m\.tech|mba|bsc|msc)\b", text, re.IGNORECASE)
    branch_match = re.search(r"\b(cse|ece|eee|mech|civil|it|ai\s*ml|ds|cs|ce|eie|aids)\b", text, re.IGNORECASE)
    return {
        "roll_number": roll_match.group(1) if roll_match else None,
        "year": year_match.group(0) if year_match else None,
        "course": course_match.group(0) if course_match else None,
        "branch": branch_match.group(0).upper() if branch_match else None
    }

def validate_text(text, expected_user_id):
    results = {
        "user_id_match": expected_user_id.lower() in text
    }

    name_match = re.search(
        r"name\s*[:\-]?\s*([A-Za-z]+(?:\s+[A-Za-z]+){1,4})(?=\s+(roll\s*no|age|dob|gender|address|phone|email|$))",
        text,
        re.IGNORECASE
    )
    results["name_extracted"] = name_match.group(1).strip() if name_match else None
    results.update(validate_mandatory_fields(text))
    results["college_found"], results["matched_college"] = validate_college_name(text)
    return results

def check_face_presence(image):
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

def run_text_validation(base64_string, user_id, ocr_confidence_threshold=40):
    status = "success"
    message = "Validation completed"
    validation = {}
    extracted_text = ""
    is_fake = False
    ocr_confidence = 0.0
    try:
        image = decode_base64_image(base64_string)
        if image is None or image.width == 0 or image.height == 0:
            raise ValueError("Invalid image data or zero dimensions")
    except Exception as e:
        return {
            "status": "error",
            "message": "Invalid image data",
            "user_id": user_id,
            "validation": {},
            "extracted_text": "",
            "is_fake": True,
            "ocr_confidence": 0.0
        }

    raw_text, avg_confidence = extract_text_with_confidence(image)
    extracted_text = raw_text
    cleaned_text = preprocess_text(raw_text)
    ocr_confidence = avg_confidence / 100 if avg_confidence else 0.0

    if avg_confidence < ocr_confidence_threshold or len(cleaned_text) < 10:
        return {
            "status": "failed",
            "message": "OCR failed or confidence too low - Classified as FAKE",
            "user_id": user_id,
            "validation": {},
            "extracted_text": extracted_text,
            "is_fake": True,
            "ocr_confidence": ocr_confidence
        }

    validation = validate_text(cleaned_text, user_id)
    validation["face_photo_found"] = check_face_presence(image)
    validation["ocr_confidence"] = ocr_confidence

    return {
        "status": status,
        "message": message,
        "user_id": user_id,
        "validation": validation,
        "extracted_text": extracted_text,
        "is_fake": not validation["face_photo_found"],  # Mark as fake if no face
        "ocr_confidence": ocr_confidence
    }
