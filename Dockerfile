FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --uid 10001 --no-create-home predictor \
    && install -d -o predictor /data
COPY predictor ./predictor

USER predictor
EXPOSE 8000
ENTRYPOINT ["python", "-m", "predictor"]
CMD ["--data", "/data/premier_league.csv", "serve", "--host", "0.0.0.0", "--db", "/data/scoreboard.sqlite3", "--schedule", "/data/fixtures_2026_27.csv"]
