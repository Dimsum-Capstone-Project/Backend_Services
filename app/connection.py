from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = "mysql+pymysql://root:admin1234@34.128.99.130:3306/palm"  # Update with your MySQL database URL

engine = create_engine(os.getenv("DATABASE_URL", DATABASE_URL))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
