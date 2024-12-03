from sqlalchemy import Column, String, Integer, Boolean, Text, ForeignKey, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
import uuid
from datetime import datetime  # Add this import

Base = declarative_base()

class User(Base):
    __tablename__ = "User"
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255))
    email = Column(String(255), unique=True)
    password_hash = Column(String(255))

class Profile(Base):
    __tablename__ = "Profile"
    profile_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("User.user_id"))
    job_title = Column(String(255))
    company = Column(String(255))
    bio = Column(Text)
    profile_picture = Column(String(255))

class ContactInfo(Base):
    __tablename__ = "Contact_Info"
    contact_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("User.user_id"))
    contact_type = Column(String(255))
    contact_value = Column(String(255))
    notes = Column(Text)

class PalmRecognitionActivity(Base):
    __tablename__ = "Palm_Recognition_Activity"
    recognition_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("User.user_id"))
    scanned_user_id = Column(String(36))
    recognition_status = Column(Boolean)
    time_scanned = Column(TIMESTAMP)

class Analytics(Base):
    __tablename__ = "Analytics"
    analytics_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("User.user_id"))
    
    # i scanned
    total_i_scanned = Column(Integer)
    successful_i_scanned = Column(Integer)
    failed_i_scanned = Column(Integer)
    last_time_i_scanned = Column(TIMESTAMP)
    
    # who scanned me
    total_whos_scanned_me = Column(Integer)
    successful_whos_scanned_me = Column(Integer)
    failed_whos_scanned_me = Column(Integer)
    last_time_whos_scanned_me = Column(TIMESTAMP)

class PasswordReset(Base):
    __tablename__ = "Password_Reset"
    reset_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("User.user_id"))
    reset_token = Column(String(255))
    token_expiration = Column(TIMESTAMP)
    is_used = Column(Boolean)

class TokenBlacklist(Base):
    __tablename__ = "Token_Blacklist"
    id = Column(Integer, primary_key=True, index=True)
    token = Column(String(255), unique=True, index=True)
    blacklisted_on = Column(TIMESTAMP, default=datetime.utcnow)