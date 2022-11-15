# Build the docker image for the project
ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

# install dependencies
RUN pip install nvtx
RUN pip install datasets
RUN pip install transformers==4.12.0