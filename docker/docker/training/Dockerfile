# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Create user and install sudo
RUN useradd -m dyalyt -s /bin/bash -g root -G sudo -u 1000

ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN echo "dyalyt:dyalyt"| chpasswd
RUN usermod -aG sudo dyalyt
RUN adduser dyalyt sudo
RUN apt update

# RUN apt upgrade -y
RUN apt install sudo libgl1-mesa-glx mesa-utils libglapi-mesa libqt5gui5  -y

RUN python -m pip install --upgrade pip
ENV PATH="${PATH}:/home/dyalyt/.local/bin"

RUN mkdir /.cache
COPY requirements.txt /.cache

USER dyalyt

# Install python requirements
RUN pip install -r /.cache/requirements.txt

WORKDIR /workspace
ENTRYPOINT /bin/bash
