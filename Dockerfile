FROM nialljb/cuda-base:latest
# FROM nialljb/cuda-mamba:latest

# Setup environment for Docker image
# configure local app structure
# ENV HOME=/root/
# ENV APPDIR="/workspace"

WORKDIR /workspace
RUN mkdir -p $WORKDIR/input
RUN mkdir -p $WORKDIR/output
# RUN mkdir -p $APPDIR/work

# Ensure pip is upgraded and install `packaging`
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging 

# Install PyTorch separately before requirements.txt
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=SimpleITK SimpleITK
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the directory the Dockerfile is into the working directory of the to be container
COPY ./ /workspace

# Configure entrypoint
ENTRYPOINT ["python3", "/workspace/inference.py"]