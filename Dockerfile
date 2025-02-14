FROM python:3.11-slim

# Setup environment for Docker image
ENV HOME=/root/
ENV FLYWHEEL="/flywheel/v0"
WORKDIR $FLYWHEEL
RUN mkdir -p $FLYWHEEL/input

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the directory the Dockerfile is into the working directory of the to be container
COPY ./ $FLYWHEEL/

# add to requirements.txt
RUN pip3 install flywheel-gear-toolkit && \
    pip3 install flywheel-sdk && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    cp license.txt /usr/local/freesurfer/.license

# Configure entrypoint
RUN bash -c 'chmod +rx $FLYWHEEL/run.py' && \
    bash -c 'chmod +rx $FLYWHEEL/app/'

ENTRYPOINT ["bash", "/flywheel/v0/start.sh"] 