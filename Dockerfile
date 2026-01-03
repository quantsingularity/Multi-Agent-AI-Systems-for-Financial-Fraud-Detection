FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY code/requirements.txt /app/requirements.txt

# Install Python dependencies (simplified - core only due to space constraints)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    scipy==1.11.1 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    pyyaml==6.0.1 \
    pytest==7.4.0

# Copy code
COPY code/ /app/code/
COPY data/ /app/data/
COPY results/ /app/results/
COPY figures/ /app/figures/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV RANDOM_SEED=42

# Create directories
RUN mkdir -p /app/data /app/results/metrics /app/results/logs /app/figures

# Default command
CMD ["python", "code/scripts/run_experiment_simple.py"]

# Labels
LABEL maintainer="[Your Email]"
LABEL description="Multi-Agent Fraud Detection System"
LABEL version="1.0.0"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sklearn, pandas, numpy; print('OK')" || exit 1
