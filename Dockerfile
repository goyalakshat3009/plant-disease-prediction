
# Use Python 3.11
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY class_indices.json .
COPY trained_model/ ./trained_model/

# Render web service port
EXPOSE 10000

# Start Streamlit on Render's assigned port
CMD ["sh", "-c", "streamlit run main.py --server.address=0.0.0.0 --server.port=${PORT:-10000} --server.headless=true"]
