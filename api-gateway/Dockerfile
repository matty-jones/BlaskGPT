FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements files and install Python dependencies
# API Gateway needs dependencies from all services it imports
COPY api-gateway/requirements.txt /app/api-gateway-requirements.txt
COPY ha-client/requirements.txt /app/ha-client-requirements.txt
COPY search-service/requirements.txt /app/search-service-requirements.txt
COPY automation-variations/requirements.txt /app/automation-variations-requirements.txt

# Install all dependencies (pip will handle duplicates)
RUN pip install --no-cache-dir \
    -r /app/api-gateway-requirements.txt \
    -r /app/ha-client-requirements.txt \
    -r /app/search-service-requirements.txt \
    -r /app/automation-variations-requirements.txt

# Copy all application code
COPY . /app/

# Set Python path
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8080

# Run the API gateway with auto-reload enabled for development
CMD ["python", "-m", "uvicorn", "api-gateway.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
