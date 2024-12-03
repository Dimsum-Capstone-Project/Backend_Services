# Capstone Project

## Overview
This project is a FastAPI-based application for palm recognition. It includes user authentication, profile management, and palm recognition functionalities.

## Features
- User Registration and Login
- Palm Recognition
- Profile Management
- Contact Information Management
- Analytics for Palm Recognition Activities

## Requirements
- Python 3.12
- MySQL Database
- Google Cloud Storage (optional)

## Setup

### Clone the Repository
```bash
git clone https://github.com/mnyasin26/capstone-project.git
```

### Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Add Model
Place your model file in the `app/ml_utils/model` directory.

### Environment Variables
Create a `.env` file in the root directory and update it with your configuration:
```properties
# Database Configuration
DATABASE_URL=mysql+pymysql://user:password@db_host:db_port/palm
FLUSH_DB=false

# JWT Configuration
ACCESS_TOKEN_EXPIRE_MINUTES=30

# GCS Configuration
USE_GCS=false
GCS_BUCKET_NAME=your_bucket_name_only_used_if_USE_GCS_true

# Palm Detection Configuration
PALM_THRESHOLD=0.8
```

### Run the Application
```bash
uvicorn main:app --host 0.0.0.0 --port 9000
```
### Notes
- The `FLUSH_DB` environment variable is used to flush the database on startup. Set it to `true` to flush the database. Make it true only for the first time.

## Usage

### Endpoints
Refer to the `List Endpoint.txt` file for a list of available API endpoints.

### Example Requests

#### Register a New User
```bash
curl -X POST "http://localhost:9000/api/v1/register" -F "email=user@example.com" -F "username=user" -F "password=secret" -F "palm_image=@path_to_image.jpg"
```

#### Login with Palm Image
```bash
curl -X POST "http://localhost:9000/api/v1/login/palm" -F "palm_image=@path_to_image.jpg"
```

## Deployment
To deploy the application on Google App Engine, update the `app.yaml` file and run:
```bash
gcloud app deploy
```

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.