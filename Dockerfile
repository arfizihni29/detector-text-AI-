FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt flask torch transformers

# Copy the rest of the app
COPY . .

# Set up user to run the app (Hugging Face Spaces requires this for security)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Expose port
EXPOSE 7860

# Run the API
CMD ["python", "api.py"]
