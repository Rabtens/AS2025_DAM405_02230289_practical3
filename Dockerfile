# Small, official slim base image keeps the final image size down
FROM python:3.11-slim

WORKDIR /app

# Copy only the dependency manifest first. Docker caches this layer,
# so as long as requirements.txt doesn't change, later code edits
# won't force a slow pip re-install on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and trained model
COPY . .

# Create a dedicated non-root user and hand ownership of /app to it.
# Running as non-root limits the blast radius if the container is compromised.
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]
