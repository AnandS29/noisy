# Based on OpenAI's mujoco-py Dockerfile

# base stage contains just binary dependencies.
# This is used in the CI build.
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    net-tools \
    parallel \
    python3.8 \
    python3.8-dev \
    python3-pip \
    rsync \
    software-properties-common \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    patchelf  \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV PATH="/venv/bin:$PATH"
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install MuJoCo
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Run Xdummy mock X server by default so that rendering will work.
COPY ci/xorg.conf /etc/dummy_xorg.conf
COPY ci/Xdummy-entrypoint.py /usr/bin/Xdummy-entrypoint.py
ENTRYPOINT ["/usr/bin/Xdummy-entrypoint.py"]

# python-req stage contains Python venv, but not code.
# It is useful for development purposes: you can mount
# code from outside the Docker container.
FROM base as python-req

WORKDIR /noisy
# Copy over just setup.py and dependencies (__init__.py and README.md)
# to avoid rebuilding venv when requirements have not changed.
COPY ./setup.py ./setup.py
COPY ./README.md ./README.md
COPY ./src/imitation/__init__.py ./src/imitation/__init__.py
COPY ci/build_and_activate_venv.sh ./ci/build_and_activate_venv.sh
COPY ./.git ./.git
RUN    ci/build_and_activate_venv.sh /venv \
    && rm -rf $HOME/.cache/pip

# full stage contains everything.
# Can be used for deployment and local testing.
# FROM python-req as full

# Delay copying (and installing) the code until the very end
# COPY . /noisy
# # Build a wheel then install to avoid copying whole directory (pip issue #2195)
# RUN python3 setup.py sdist bdist_wheel
# RUN pip install -e .

# # Default entrypoints
# # python3 test_pref.py --env linear1d --pref --stats --noise --verbose --timesteps 25000 --comparisons 90000 --algo trpo
# CMD python3 test_pref.py --env linear1d --pref --stats --noise --verbose --timesteps 25000 --comparisons 90000 --algo trpo
