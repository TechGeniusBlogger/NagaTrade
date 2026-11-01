# Kronos Forecast, hosted (Render). Runs the Flask app on $PORT behind a password (APP_PASSWORD).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KRONOS_HOME=/data \
    HOST=0.0.0.0 \
    PORT=10000

WORKDIR /app
COPY app_source/requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app_source/ .
# Bake the verified model into the image (KRONOS_HOME/models).
RUN python fetch_model.py

EXPOSE 10000
CMD ["python", "kronos_app.py", "--no-browser"]
