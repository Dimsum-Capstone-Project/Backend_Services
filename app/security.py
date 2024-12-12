# FILE: security.py
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt
from typing import Optional
from sqlalchemy.orm import Session
from app.models import TokenBlacklist  # Assuming you have a TokenBlacklist model defined

# Create a CryptContext object
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "admin1234"  # Use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def invalidate_token(token: str, db: Session):
    blacklisted_token = TokenBlacklist(token=token)
    db.add(blacklisted_token)
    db.commit()