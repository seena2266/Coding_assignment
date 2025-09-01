# Base image with Python 3.10 (stable for ML libraries)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (needed for numpy, pandas, faiss, pillow, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    vim \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first (to leverage Docker caching)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install -r requirements.txt

# Copy the project code into the container
COPY . .

# Expose ports for Streamlit and Jupyter
EXPOSE 8501 8888

# Default command: Streamlit app inside rag/ folder
CMD ["streamlit", "run", "drivelm_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
