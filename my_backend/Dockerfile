FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory with proper permissions
RUN mkdir -p /app/temp_uploads && chmod 777 /app/temp_uploads

# Set environment variables
ENV PORT=5001
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV GUNICORN_CMD_ARGS="--workers=1 --threads=8 --timeout=0 --bind=:${PORT} --log-level=info"

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run with gunicorn
CMD ["gunicorn", "app:app"]