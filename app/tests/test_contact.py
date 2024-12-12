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