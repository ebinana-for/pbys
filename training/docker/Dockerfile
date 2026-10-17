FROM nvcr.io/nvidia/pytorch:24.09-py3

# Install some libs
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -qq && apt-get install --no-install-recommends ffmpeg libsm6 libxext6 libmagickwand-dev -y

# remove old opencv 4.7.0 installed by default
RUN pip uninstall -y opencv
RUN rm -r /usr/local/lib/python3.10/dist-packages/cv2

# Install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip wheel setuptools
RUN pip --no-cache-dir install -r /tmp/requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/workspace"

WORKDIR /workspace
