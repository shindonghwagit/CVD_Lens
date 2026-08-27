FROM python:3.11-slim
WORKDIR /app
# ffmpeg: /infer/video 프레임 인코딩에 필요 (없으면 영상 보정이 검은 화면으로 실패)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY config.py .
COPY guided.py .
COPY model/ ./model/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
