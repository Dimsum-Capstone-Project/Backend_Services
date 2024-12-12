from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Response
from jose import JWTError, jwt
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.connection import get_db
from app.models import User, PasswordReset, Profile
from app.security import hash_password, verify_password, create_access_token, invalidate_token
from app.schemas import RegisterRequest, LoginRequest, LogoutRequest, PasswordResetRequest, PasswordResetConfirm
from app.email_utils import send_reset_email  # Assuming you have a utility to send emails
from app.ml_utils.ml_utils import process_palm_image, convert_to_jpg_and_return
from app.ml_utils.preprocessing.palm_processor_enhanced import PalmPreprocessor
from app.routers import recognizer
import os
import logging
import json
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv(override=True)

router = APIRouter()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize preprocessor
preprocessor = PalmPreprocessor(target_size=(128, 128))

# Setup folders
base_dir = "app/ml_utils/data"
raw_login_dir = os.path.join(base_dir, "raw/login_attempt")
raw_register_dir = os.path.join(base_dir, "raw/register_attempt")
raw_registered_dir = os.path.join(base_dir, "raw/registered")

aug_login_dir = os.path.join(base_dir, "aug/login_attempt")
aug_registered_dir = os.path.join(base_dir, "aug/registered")


if os.getenv("USE_GCS", "false") == "false":
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(raw_login_dir, exist_ok=True)
    os.makedirs(raw_register_dir, exist_ok=True)
    os.makedirs(raw_registered_dir, exist_ok=True)
    os.makedirs(aug_login_dir, exist_ok=True)
    os.makedirs(aug_registered_dir, exist_ok=True)


# os.makedirs(raw_dir, exist_ok=True)
# os.makedirs(aug_dir, exist_ok=True)


# Remove recognizer initialization
# recognizer = PalmPrintRecognizer(
#     model_path="app/ml_utils/model/v2/palm_print_siamese_model.h5"
# )

# test env
print("ENVIRONMENT VARIABLES")
print(os.getenv("USE_GCS", "false"))

"""
Registers a new user.
This endpoint allows a new user to register by providing their email, username, and password.
The password is hashed before storing it in the database.
Args:
    request (RegisterRequest): The registration request containing user details.
    db (Session, optional): The database session dependency.
Returns:
    dict: A dictionary containing a success message and the registered user's email and username,
          or an error message if the registration fails.
"""
@router.post("/register")
async def register(
    palm_image: UploadFile = File(...),
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    email = email
    username = username
    password = password

    try:
        # Process uploaded image
        img, img_path, user_id = await process_palm_image(palm_image, raw_register_dir)

        # check if the email is already in use
        user = db.query(User).filter(User.email == email).first()
        if user:
            return Response(
                content=json.dumps({"message": "Email already in use"}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )

        # Preprocess image
        processed_image, notes = preprocessor.preprocess_image(img)
        print("Notes")
        print(notes)


        if processed_image is None:
            return Response(
                content=json.dumps({"message": "Failed to preprocess palm image could not recognize any hand, please try again using a different image", "notes": notes}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )

        # Generate augmentations
        augmented_images = preprocessor.generate_augmentations(processed_image)
        
        # consistency step
        first_augmented_image = list(augmented_images.values())[0]
        first_augmented_image = convert_to_jpg_and_return(first_augmented_image)
        
        # check if the image is already in the database
        recognizer.load_database("app/ml_utils/data/palm_print_db.json")  # Load the database
        print(os.getenv("PALM_THRESHOLD", 290))
        result_id, result_similarity = recognizer.find_match3(first_augmented_image, threshold=float(os.getenv("PALM_THRESHOLD", 290)))  # Find match for the processed image
        recognizer.reset_database()  # Reset the database

        if result_id:
            return Response(
                content=json.dumps({"message": "Palm print already registered"}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )

        saved_paths = preprocessor.save_augmented_images(augmented_images, base_dir=aug_registered_dir, user_id=user_id)

        for img_temp in list(augmented_images.values()):
            proccessed_img = convert_to_jpg_and_return(img_temp)
            recognizer.add_to_database(user_id, proccessed_img)

        print("Database updated")
        print("Finding match")
        print(saved_paths[0])

        # PALM_THRESHOLD=0.8
        result_id, result_similarity = recognizer.find_match3(first_augmented_image, threshold=float(os.getenv("PALM_THRESHOLD", 290)))  # Find match for the processed

        if result_id is None:
            return Response(
                content=json.dumps({"message": "Registration failed, something wrong, please try again"}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )

        # convert numpy float to python float
        result_similarity = result_similarity.item()
        result_similarity = str (result_similarity)
        
        print("Match found")
        print(result_id)
        print(result_similarity)

        # recognizer.append_database("app/ml_utils/data/palm_print_db.json")  # Save the updated database
        recognizer.append_database("app/ml_utils/data/palm_print_db.json")  # Save the updated database
        recognizer.reset_database()  # Reset the database

    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return Response(
            content=json.dumps({"detail": str(e)}),
            status_code=500,
            media_type="application/json"
        )

    # hash the password
    password = hash_password(password)

    # Create a new user instance
    new_user = User(user_id=user_id,email=email, name=username, password_hash=password)  # Hash the password before saving

    # add the new user to the database
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
    except Exception as e:  
        db.rollback()
        return Response(
            content=json.dumps({"error": str(e)}),
            status_code=500,
            media_type="application/json"
        )
    
    # add profile
    new_profile = Profile(user_id=new_user.user_id)

    try:
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
    except Exception as e:
        db.rollback()
        return {"error": str(e)}

    # save image img
    preprocessor.save_image(img, os.path.join(raw_registered_dir, f"{user_id}.png"))

    # return the user details
    return Response(
        content=json.dumps({"message": "Registration successful", "email": email, "username": username}),
        status_code=status.HTTP_201_CREATED,
        media_type="application/json"
    )

@router.post("/login/palm")
async def login_palm(
    palm_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        # Process uploaded image
        img, img_path, image_id = await process_palm_image(palm_image, raw_login_dir)

        # Preprocess image
        processed_image, notes = preprocessor.preprocess_image(img)
        print("Notes")
        print(notes)

        if processed_image is None:
            return Response(
                content=json.dumps({"message": "Failed to preprocess palm image could not recognize any hand, please try again using a different image", "notes": notes}),
                status_code=status.HTTP_400_BAD_REQUEST,
                media_type="application/json"
            )
        
        # save the processed image on temp folder
        # temp_path = "app/ml_utils/data/temp"

        # if os.getenv("USE_GCS", "false") == "false":
        #     os.makedirs(temp_path, exist_ok=True)

        temp_image_path = os.path.join(aug_login_dir, f"{image_id}.jpg")
        
        # preprocessor.save_image(processed_image, temp_image_path)
        preprocessor.save_image(processed_image, temp_image_path)

        # return Response(
        #     content=json.dumps({"message": "Login successful", "image_id": image_id}),
        #     status_code=status.HTTP_200_OK,
        #     media_type="application/json"
        # )
        print("Finding match")

        proccessed_img = convert_to_jpg_and_return(processed_image)

        recognizer.load_database("app/ml_utils/data/palm_print_db.json")  # Load the database
        result_id, best_similarity = recognizer.find_match3(proccessed_img, threshold=float(os.getenv("PALM_THRESHOLD", 290)), use_threshold=False)  # Find match for the processed image
        recognizer.reset_database()  # Reset the database

        # convert numpy float to python float
        # minimum_distance = minimum_distance.item()
        # minimum_distance = str (minimum_distance)

        print("Match found")
        print(result_id)
        print(best_similarity)

        if result_id is None:
            return Response(
                content=json.dumps({"message": "Login failed, palm print not recognized, please try again"}),
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json"
            )

        user = db.query(User).filter(User.user_id == result_id).first()
        
        if not user:
            return Response(
                content=json.dumps({"message": "Login failed, user not found"}),
                status_code=status.HTTP_401_UNAUTHORIZED,
                media_type="application/json"
            )
        
        access_token_expires = timedelta(minutes=float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 30)))
        access_token = create_access_token(
            data={"sub": user.email}, expires_delta=access_token_expires
        )
        # recognizer.append_database("app/ml_utils/data/palm_print_db.json")  # Save the updated database

        return Response(
            content=json.dumps({"message": "Login successful", "distance": f"{best_similarity:.2f}", "token": {"access_token": access_token, "token_type": "bearer"}}),
            status_code=status.HTTP_200_OK,
            media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return Response(
            content=json.dumps({"detail": str(e)}),
            status_code=500,
            media_type="application/json"
        )

@router.post("/login")
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        return Response(
            content=json.dumps({"message": "Incorrect email or password"}),
            status_code=status.HTTP_401_UNAUTHORIZED,
            media_type="application/json",
            headers={"WWW-Authenticate": "Bearer"}
        )

    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )

    return Response(content=json.dumps({
        "access_token": access_token,
        "token_type": "bearer"
    }), status_code=200, media_type="application/json")

@router.post("/logout")
async def logout(request: LogoutRequest, db: Session = Depends(get_db)):
    try:
        invalidate_token(request.token, db)
        return Response(
            content=json.dumps({"message": "Logout successful"}),
            status_code=status.HTTP_200_OK,
            media_type="application/json"
        )
    except Exception as e:
        return Response(
            content=json.dumps({"message": "Invalid token"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

@router.post("/password_reset")
async def password_reset(request: PasswordResetRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == request.email).first()
    if not user:
        return Response(
            content=json.dumps({"message": "User not found"}),
            status_code=status.HTTP_404_NOT_FOUND,
            media_type="application/json"
        )
    

    reset_token = create_access_token(data={"sub": user.email}, expires_delta=timedelta(hours=1))
    password_reset = PasswordReset(
        user_id=user.user_id,
        reset_token=reset_token,
        token_expiration=datetime.utcnow() + timedelta(hours=1),
        is_used=False
    )

    db.add(password_reset)
    db.commit()

    send_reset_email(user.email, reset_token)  # Send the reset email

    return Response(
        content=json.dumps({"message": "Password reset email sent", "reset_token": reset_token}),
        status_code=status.HTTP_200_OK,
        media_type="application/json"
    )

@router.post("/password_reset/confirm")
async def password_reset_confirm(request: PasswordResetConfirm, db: Session = Depends(get_db)):
    password_reset = db.query(PasswordReset).filter(PasswordReset.reset_token == request.token).first()
    if not password_reset or password_reset.is_used or password_reset.token_expiration < datetime.utcnow():
        return Response(
            content=json.dumps({"message": "Invalid or expired token"}),
            status_code=status.HTTP_400_BAD_REQUEST,
            media_type="application/json"
        )

    user = db.query(User).filter(User.user_id == password_reset.user_id).first()
    if not user:
        return Response(
            content=json.dumps({"message": "User not found"}),
            status_code=status.HTTP_404_NOT_FOUND,
            media_type="application/json"
        )

    user.password_hash = hash_password(request.new_password)
    password_reset.is_used = True

    db.commit()

    return Response(
        content=json.dumps({"message": "Password reset successful"}),
        status_code=status.HTTP_200_OK,
        media_type="application/json"
    )