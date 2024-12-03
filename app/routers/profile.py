from fastapi import APIRouter, Depends, HTTPException, status, File, Form, UploadFile
from sqlalchemy.orm import Session
from app.schemas import EditProfileRequest
from app.connection import get_db
from app.dependencies import get_current_user
from app.models import User, Profile
import uuid
import os
from fastapi.responses import Response
import json
from dotenv import load_dotenv
from app.storage_utils import upload_to_gcs, get_from_gcs

load_dotenv()

router = APIRouter()

@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Get profile logic using current_user
    profile = db.query(Profile).filter(Profile.user_id == current_user.user_id).first()
    if not profile:
        return Response(
            content=json.dumps({"detail": "Profile not found"}),
            status_code=status.HTTP_404_NOT_FOUND,
            media_type="application/json"
        )
    return Response(content=json.dumps({
        "email": current_user.email,
        "username": current_user.name,
        "bio": profile.bio,
        "company": profile.company,
        "job_title": profile.job_title,
        "profile_picture": profile.profile_picture
    }), status_code=200, media_type="application/json")

@router.post("/profile/edit")
async def edit_profile(
    name: str = Form(...),
    bio: str = Form(...),
    company: str = Form(...),
    job_title: str = Form(...),
    profile_picture: UploadFile = File(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # Edit profile logic using current_user
    user = db.query(User).filter(User.user_id == current_user.user_id).first()
    profile = db.query(Profile).filter(Profile.user_id == current_user.user_id).first()

    if not user:
        return Response(
            content=json.dumps({"detail": "User not found"}),
            status_code=status.HTTP_404_NOT_FOUND,
            media_type="application/json"
        )
    
    if not profile:
        return Response(
            content=json.dumps({"detail": "Profile not found"}),
            status_code=status.HTTP_404_NOT_FOUND,
            media_type="application/json"
        )
    
    # Update user and profile details
    user.name = name
    profile.bio = bio
    profile.company = company
    profile.job_title = job_title

    if profile_picture:
        _, file_extension = os.path.splitext(profile_picture.filename)
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        profile.profile_picture = unique_filename

        if os.getenv("USE_GCS", "false") == "true":
            upload_to_gcs(unique_filename, profile_picture.file.read(), "dimsum_palm_public")
        else:
            with open(f"uploads/{unique_filename}", "wb") as buffer:
                buffer.write(profile_picture.file.read())

    try:
        db.commit()
        db.refresh(user)
        db.refresh(profile)
    except Exception as e:
        db.rollback()
        return Response(
            content=json.dumps({"detail": str(e)}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    return Response(content=json.dumps({
        "message": "Profile updated successfully",
        "user": {
            "email": user.email or "",
            "username": user.name or ""
        },
        "profile": {
            "bio": profile.bio or "",
            "company": profile.company or "",
            "job_title": profile.job_title or "",
            "profile_picture": profile.profile_picture or ""
        }
    }), status_code=200, media_type="application/json")





