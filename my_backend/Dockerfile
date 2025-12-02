# Minimal Dockerfile for fast deploymentt
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chunk_uploads temp_uploads uploads/file_uploads

# Expose port
EXPOSE 8080

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with proper configuration for Cloud Run - Production Streaming Optimized
# -k gthread: bolje za streaming od sync workera
# -w 1: MORA biti 1 worker za Socket.IO bez Redis-a (session sharing)
# --threads 8: više threadova za paralelizam umjesto više workera
# --timeout 900: 15 min za velike fajlove
# --graceful-timeout 30: graceful shutdown
# --keep-alive 75: keep connection alive za streaming
CMD ["gunicorn", "-k", "gthread", "-w", "1", "--threads", "8", "-b", "0.0.0.0:8080", "--timeout", "900", "--graceful-timeout", "30", "--keep-alive", "75", "app:app"]