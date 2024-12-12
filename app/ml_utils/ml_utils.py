from fastapi import UploadFile, HTTPException
import numpy as np
import cv2
import os
from datetime import datetime
import uuid
from app.ml_utils.preprocessing.palm_processor_enhanced import PalmPreprocessor
import logging
from app.storage_utils import upload_to_gcs, get_from_gcs
from dotenv import load_dotenv

load_dotenv(override=True)


print("Hello World")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_palm_image(file: UploadFile, file_path) -> tuple:
    """Process uploaded palm image"""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # uniform limit the width only
        # img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Save original image
        user_id = str(uuid.uuid4())
        original_path = os.path.join(file_path, f"{user_id}.jpg")
        
        if os.getenv("USE_GCS", "false") == "true":
            contents = cv2.imencode('.jpg', img)[1].tobytes()
            upload_to_gcs(original_path, contents)
            
        else:
            cv2.imwrite(original_path, img)


        # upload_to_gcs("dimsum_palm_private", f"{user_id}.jpg", img)

        return img, original_path, user_id

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

def convert_to_jpg_and_return(image):
    content = cv2.imencode('.jpg', image)[1].tobytes()
    preps = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)
    return preps