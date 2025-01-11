from fastapi import FastAPI, UploadFile, HTTPException, File
import logging
from app.model.model import (
    get_tuberculosis_model, get_pneumonia_model, get_covid19_model, predict_with_model
)
# Maximum file size (10MB limit)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Define class names
TUBERCULOSIS_CLASSES = ["No Turberculosis", "Turberculosis Detected"]
PNEUMONIA_CLASSES = ["No Pneumonia", "Pneumonia Detected"]
COVID19_CLASSES = ["Negative for COVID-19", "Positive for COVID-19"]

app = FastAPI()

@app.post("/predict_pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    """Predict Pneumonia from X-ray image."""
    try:
        file_content = await file.read()
        logging.info(f"File size: {len(file_content)} bytes")

        model = get_pneumonia_model()
        result = predict_with_model(model, file_content, PNEUMONIA_CLASSES)

        return {
            "filename": file.filename,
            "confidence": result["confidence"],
            "description": result["description"]
        }
    except HTTPException as e:
        logging.error(f"Pneumonia prediction error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected Pneumonia prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict_tuberculosis")
async def predict_tuberculosis(file: UploadFile = File(...)):
    """Predict Tuberculosis from X-ray image."""
    try:
        file_content = await file.read()
        logging.info(f"File size: {len(file_content)} bytes")

        model = get_tuberculosis_model()
        result = predict_with_model(model, file_content, TUBERCULOSIS_CLASSES)

        return {
            "filename": file.filename,
            "confidence": result["confidence"],
            "description": result["description"]
        }
    except HTTPException as e:
        logging.error(f"Tuberculosis prediction error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected Tuberculosis prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/predict_covid19")
async def predict_covid19(file: UploadFile = File(...)):
    """Predict COVID-19 from X-ray image."""
    try:
        file_content = await file.read()
        logging.info(f"File size: {len(file_content)} bytes")

        model = get_covid19_model()
        result = predict_with_model(model, file_content, COVID19_CLASSES)

        return {
            "filename": file.filename,
            "confidence": result["confidence"],
            "description": result["description"]
        }
    except HTTPException as e:
        logging.error(f"COVID-19 prediction error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected COVID-19 prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    