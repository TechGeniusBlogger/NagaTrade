# Kronos Forecast, hosted (Render). Runs the Flask app on $PORT behind a password (APP_PASSWORD).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    KRONOS_HOME=/data \
    HOST=0.0.0.0 \
    PORT=10000 \
    KRONOS_INT8=1 \
    KRONOS_CHUNK=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app
COPY app_source/requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY app_source/ .
# Bake the verified model into the image (KRONOS_HOME/models).
RUN python fetch_model.py
# 8-bit copy of the model so it fits a 512 MB instance (KRONOS_INT8=1); KRONOS_CHUNK=1 runs the
# 30 paths one at a time to keep the working memory small. Raise KRONOS_CHUNK on a bigger plan.
RUN python quantize_model.py

EXPOSE 10000
CMD ["python", "kronos_app.py", "--no-browser"]
