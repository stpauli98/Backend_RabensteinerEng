# Use Python slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY my_backend/requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY my_backend/ ./

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Set environment variable for Flask
ENV PORT=8080

# Run the application
CMD ["python", "app.py"]