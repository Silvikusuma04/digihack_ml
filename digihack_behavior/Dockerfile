FROM python:3.13-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglx0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /code

# Copy requirements.txt first to leverage Docker's cache
COPY ./requirements.txt /code/requirements.txt

# Create a virtual environment
RUN python -m venv /code/venv

# Install dependencies within the virtual environment
RUN /code/venv/bin/pip install --upgrade pip
RUN /code/venv/bin/pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the rest of the application code
COPY . /code/

# Command to run the application
CMD ["/code/venv/bin/python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30018"]
