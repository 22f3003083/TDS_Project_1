FROM python:3.12-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    libssl-dev \
    libffi-dev

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Install required Python packages
RUN pip install fastapi uvicorn requests python-dateutil pandas scipy python-dotenv httpx markdown duckdb numpy Pillow openai

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin:$PATH"

# Set up the application directory
WORKDIR /app

# Copy the combined application file
COPY app.py /app

# Expose port 8000 for the FastAPI application
EXPOSE 8000

# Run the application using uvicorn via uv binary
CMD ["/root/.local/bin/uv", "run", "app.py"]
