# 1. Base Image: NVIDIA CUDA with Python support (Ubuntu 22.04, CUDA 12.1, Python 3.10)
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 2. Set Environment Variables
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
# Sets debian frontend to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive
# Set a default timezone to prevent tzdata configuration prompts
ENV TZ=Etc/UTC
# Set NLTK data path so NLTK knows where to find 'punkt' downloaded system-wide
ENV NLTK_DATA=/usr/share/nltk_data
# 2. Set Environment Variables
ENV COQUI_TOS_AGREED=1

# 3. Install System Dependencies: Python, pip, ffmpeg
# Ubuntu 22.04 comes with Python 3.10. We'll ensure python3-pip is installed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg && \
    # Clean up apt caches to reduce image size
    rm -rf /var/lib/apt/lists/*

# Make python3 and pip3 the default 'python' and 'pip'
# RUN ln -s /usr/bin/python3 /usr/bin/python 
# && \ ln -s /usr/bin/pip3 /usr/bin/pip

# 4. Set up the Working Directory in the container
WORKDIR /app

RUN pip3 install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# 5. Copy requirements.txt and install Python packages
# This is done before copying the rest of the code to leverage Docker layer caching.
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 6. Pre-download NLTK 'punkt' tokenizer data to the system-wide NLTK data path
RUN python3 -m nltk.downloader -d /usr/share/nltk_data punkt

# 7. Copy the rest of your application code into the /app directory
COPY . /app/
# This includes your dts.py (or demo.py), config.py, etc.

RUN python3 /app/pre_download_models.py

# 8. (Optional) Create directories for data volumes if you want to ensure they exist
# RUN mkdir -p /data/input /data/output /data/speaker

# 9. Set the default command to execute when the container starts
# This will run your script. Arguments will be appended at runtime.
ENTRYPOINT ["python3", "dts.py"]

# Example: To show help by default if no other command is given
# CMD ["--help"]