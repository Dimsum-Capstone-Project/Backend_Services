import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base, User
from main import app
from connection import get_db
import json

DATABASE_URL = "mysql+pymysql://root:admin1234@localhost:3306/palm_test"  # Use a test database

engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency override
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="module")
def client():
    # Base.metadata.create_all(bind=engine)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Execute dummies.sql
    Session = sessionmaker(bind=engine)
    session = Session()
    with open('sql/dummies.sql', 'r') as file:
        sql_commands = file.read().split(';')
        for command in sql_commands:
            if command.strip():
                session.execute(text(command))
        session.commit()
    session.close()
    with TestClient(app) as c:
        yield c
    # Base.metadata.drop_all(bind=engine)
    

def test_register(client):
    with open("test_image/DATASET-004.jpg", "rb") as palm_image:
        response = client.post("/api/v1/register", files={"palm_image": palm_image}, data={
            "email": "john.doe@example2.com",
            "username": "johndoe",
            "password": "12345678"
        })
    print(response)
    assert response.status_code == 201  # Adjusted status code
    assert response.json()["message"] == "Registration successful"  # Adjusted message


def test_login(client):
    response = client.post("/api/v1/login", json={
        "email": "john.doe@example.com",
        "password": "12345678"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_logout(client):
    # First, log in to get the access token
    response = client.post("/api/v1/login", json={
        "email": "john.doe@example.com",
        "password": "12345678"
    })
    print(response)
    assert response.status_code == 200
    access_token = response.json()["access_token"]

    # Then, log out using the access token
    response = client.post("/api/v1/logout",headers={"Authorization": f"Bearer {access_token}"}, json={"token": access_token})
    assert response.status_code == 200
    assert response.json()["message"] == "Logout successful"

def test_password_reset(client):
    response = client.post("/api/v1/password_reset", json={
        "email": "john.doe@example.com"
    })
    assert response.status_code == 200
    assert response.json()["message"] == "Password reset email sent"

def test_password_reset_confirm(client):
    # First, request a password reset to get the token
    response = client.post("/api/v1/password_reset", json={
        "email": "john.doe@example.com"
    })
    assert response.status_code == 200
    reset_token = response.json().get("reset_token")
    assert reset_token is not None, "Reset token not found in response"

    # Now, confirm the password reset using the token
    response = client.post("/api/v1/password_reset/confirm", json={
        "token": reset_token,
        "new_password": "newtestpassword"
    })
    assert response.status_code == 200
    assert response.json()["message"] == "Password reset successful"

