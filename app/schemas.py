from pydantic import BaseModel

class RegisterRequest(BaseModel):
    email: str
    username: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LogoutRequest(BaseModel):
    token: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

class EditProfileRequest(BaseModel):
    name: str
    bio: str
    company: str
    job_title: str
    profile_picture: str

class TokenData(BaseModel):
    user_email: str

class AddContactInfoRequest(BaseModel):
    contact_type: str
    contact_value: str
    notes: str

class DeleteContactInfoRequest(BaseModel):
    contact_id: str

class EditContactInfoRequest(BaseModel):
    contact_id: str
    contact_type: str
    contact_value: str
    notes: str