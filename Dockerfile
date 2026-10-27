
# Use Python 3.11
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing bytecode
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY class_indices.json .
COPY trained_model/ ./trained_model/

# Expose Streamlit port
EXPOSE 8501

# Start the Streamlit application
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
