FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and audio
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    espeak \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Run the streamlit app
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
