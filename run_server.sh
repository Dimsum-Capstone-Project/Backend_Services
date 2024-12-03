#!/bin/bash

# Activate the virtual environment
source caps/bin/activate

# Run the uvicorn server
uvicorn app.main:app --host 0.0.0.0 --port 9000