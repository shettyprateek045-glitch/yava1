# ShapeVision Studio — Computer Vision Shape Classification Pipeline
# Serves the FastAPI control center console & REST API (api.py).
#
# Build:  docker build -t shapevision-studio .
# Run:    docker run -p 8000:8000 shapevision-studio
# Open:   http://localhost:8000/        (studio UI)
#         http://localhost:8000/docs    (Swagger API docs)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-advanced.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-advanced.txt

COPY . .

# Verify pipeline integrity during build
RUN pytest tests/ -v

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
