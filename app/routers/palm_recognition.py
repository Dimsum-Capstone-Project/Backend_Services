from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from sqlalchemy.orm import Session
from app.connection import get_db
from app.models import User, Profile, ContactInfo, PalmRecognitionActivity, Analytics
from app.dependencies import get_current_user
from app.ml_utils.ml_utils import process_palm_image, convert_to_jpg_and_return
from app.ml_utils.preprocessing.palm_processor_enhanced import PalmPreprocessor
from app.routers import recognizer
from datetime import datetime
import os
import json
from dotenv import load_dotenv

load_dotenv(override=True)

router = APIRouter()

# Initialize preprocessor
preprocessor = PalmPreprocessor(target_size=(128, 128))

@router.post("/recognize_palm")
async def recognize_palm(
    palm_image: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Process uploaded image
        img, file_path, image_id = await process_palm_image(palm_image, "app/ml_utils/data/raw/recognition")
        processed_image, notes = preprocessor.preprocess_image(img)

        if processed_image is None:
            return Response(
                content=json.dumps({"message": "Failed to preprocess palm image could not recognize any hand, please try again using a different image", "notes": notes}),
                status_code=400,
                media_type="application/json"
            )

        # Save the processed image temporarily
        temp_path = "app/ml_utils/data/aug/recognition"
        if os.getenv("USE_GCS", "false") == "false":
            os.makedirs(temp_path, exist_ok=True)
        temp_image_path = os.path.join(temp_path, f"{image_id}.jpg")
        # preprocessor.save_image(processed_image, temp_image_path)
        preprocessor.save_image(processed_image, temp_image_path)

        proccessed_img = convert_to_jpg_and_return(processed_image)

        # Perform palm recognition
        recognizer.load_database("app/ml_utils/data/palm_print_db.json")
        result_id, best_similarity = recognizer.find_match3(proccessed_img, threshold=float(os.getenv("PALM_THRESHOLD", 290)), use_threshold=False)
        recognizer.reset_database()

        print(f"Best similarity: {best_similarity}")

        # Create a new PalmRecognitionActivity
        recognition_activity = PalmRecognitionActivity(
            user_id=current_user.user_id,
            scanned_user_id=result_id if result_id else None,
            recognition_status=True if result_id else False,
            time_scanned=datetime.utcnow()
        )
        db.add(recognition_activity)

        # Update Analytics for current user
        analytics = db.query(Analytics).filter(Analytics.user_id == current_user.user_id).first()
        if not analytics:
            analytics = Analytics(
                user_id=current_user.user_id,
                total_i_scanned=0,
                successful_i_scanned=0,
                failed_i_scanned=0,
                last_time_i_scanned=None
            )
            db.add(analytics)

        analytics.total_i_scanned = (analytics.total_i_scanned or 0) + 1
        analytics.last_time_i_scanned = datetime.utcnow()
        if result_id:
            analytics.successful_i_scanned = (analytics.successful_i_scanned or 0) + 1
        else:
            analytics.failed_i_scanned = (analytics.failed_i_scanned or 0) + 1

        # If a user was recognized, update their Analytics and return their info
        if result_id:
            recognized_user = db.query(User).filter(User.user_id == result_id).first()
            profile = db.query(Profile).filter(Profile.user_id == result_id).first()
            contacts = db.query(ContactInfo).filter(ContactInfo.user_id == result_id).all()

            # if the recognized user is the same as the current user, return an error
            # if recognized_user.user_id == current_user.user_id:
            #     return Response(
            #         content=json.dumps({"message": "You cannot scan yourself."}),
            #         status_code=400,
            #         media_type="application/json"
            #     )

            # Update Analytics for recognized user
            recognized_analytics = db.query(Analytics).filter(Analytics.user_id == recognized_user.user_id).first()
            if not recognized_analytics:
                recognized_analytics = Analytics(
                    user_id=recognized_user.user_id,
                    total_whos_scanned_me=0,
                    successful_whos_scanned_me=0,
                    failed_whos_scanned_me=0,
                    last_time_whos_scanned_me=None
                )
                db.add(recognized_analytics)

            recognized_analytics.total_whos_scanned_me = (recognized_analytics.total_whos_scanned_me or 0) + 1
            recognized_analytics.successful_whos_scanned_me = (recognized_analytics.successful_whos_scanned_me or 0) + 1
            recognized_analytics.last_time_whos_scanned_me = datetime.utcnow()

            db.commit()

            return Response(
                content=json.dumps({
                    "distance": f"{best_similarity:.2f}",
                    "user": {
                        "email": recognized_user.email,
                        "username": recognized_user.name
                    },
                    "profile": {
                        "bio": profile.bio or "",
                        "company": profile.company or "",
                        "job_title": profile.job_title or "",
                        "profile_picture": profile.profile_picture or ""
                    },
                    "contacts": [
                        {
                            "contact_type": contact.contact_type,
                            "contact_value": contact.contact_value,
                            "notes": contact.notes
                        } for contact in contacts
                    ]
                }),
                status_code=200,
                media_type="application/json"
            )
        else:
            db.commit()
            return Response(
                content=json.dumps({"message": "Palm print not recognized"}),
                status_code=200,
                media_type="application/json"
            )

    except Exception as e:
        db.rollback()
        return Response(
            content=json.dumps({"message": str(e)}),
            status_code=500,
            media_type="application/json"
        )