# Use Python slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt first to leverage Docker cache
COPY my_backend/requirements.txt .

# Create and activate virtual environment in container
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir -r requirements.txt

# Now copy all code from backend folder
COPY my_backend/ .

# Expose the port the app runs on
EXPOSE 8080

# Set default command to run the application
CMD ["./venv/bin/python", "app.py"]
