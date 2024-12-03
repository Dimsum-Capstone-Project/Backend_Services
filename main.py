from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.routers import auth, history, analytics, profile, palm_recognition
from app.routers import contact
from app.models import Base
from app.connection import engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
# import mediapipe as mp
import os
from dotenv import load_dotenv
from app.storage_utils import upload_to_gcs, get_from_gcs, delete_all_files_in_gcs

load_dotenv()

base_dir = "app/ml_utils/data"
raw_dir = os.path.join(base_dir, "raw")
aug_dir = os.path.join(base_dir, "aug")

app = FastAPI()

# Serve static files from the "uploads" directory
# app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
async def read_root():
    return {"userid": "Bagas!"}

app.include_router(auth.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(profile.router, prefix="/api/v1")
app.include_router(contact.router, prefix="/api/v1")
app.include_router(palm_recognition.router, prefix="/api/v1")

# Create all tables in the database
@app.on_event("startup")
async def startup_event():
    if os.getenv("FLUSH_DB", "false") == "true":
        print("Flushing database and storage...")
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

        # Flush all images in the gcs or local storage
        if os.getenv("USE_GCS", "false") == "true":
            delete_all_files_in_gcs()
            delete_all_files_in_gcs("dimsum_palm_public")
            # upload_to_gcs("dimsum_palm_private", f"{base_dir}/palm_print_db.json", "")
        else:
            for file in os.listdir(raw_dir):
                os.remove(os.path.join(raw_dir, file))


            # with open(f"{base_dir}/palm_print_db.json", "w") as f:
            #     f.write("")

    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)









