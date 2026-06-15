ARG CUDA_VERSION=12.6.3
ARG OS_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${OS_VERSION}

ARG PYTHON_VERSION=3.12

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES=0 \
    TEST_TYPE=single_gpu \
    DEVICE=cuda \
    PATH="/work/.venv/bin:$PATH" \
    HF_HOME="/workspaces/huggingface"

# Run apt update, install all necessary system libraries, register deadsnakes PPA,
# install python, openssh-server, and clean up apt cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common build-essential git curl sudo tmux neovim htop nvtop \
    libglib2.0-0 libgl1 libglx-mesa0 ffmpeg openssh-server cmake \
    libusb-1.0-0-dev speech-dispatcher libgeos-dev portaudio19-dev \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/uv \
    && useradd --create-home --shell /bin/bash user \
    && usermod -aG sudo user \
    && echo "user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers \
    && mkdir /work \
    && mkdir /workspaces \
    && mkdir /var/run/sshd \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /work

RUN chown -R user:user /work && chown -R user:user /workspaces

USER user

# Copy pyproject.toml and uv.lock first to cache dependency installations
COPY --chown=user:user pyproject.toml uv.lock ./

# Create virtual environment and install only dependencies (skip project package build)
RUN uv venv --python python${PYTHON_VERSION} && \
    uv sync --frozen --no-install-project

# Copy entrypoint script from docker/ folder
COPY --chown=user:user docker/entrypoint.sh /entrypoint.sh

# Copy the rest of the application files (will be fast due to .dockerignore and cache)
COPY --chown=user:user . /work

# Perform a final sync to install the local packages in editable mode
RUN uv sync --frozen

# Ensure activate script is loaded into the user bash session
RUN echo "source /work/.venv/bin/activate" >> /home/user/.bashrc

ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash", "-c", "sudo chown -R user:user /work; sudo chown -R user:user /workspaces; mkdir -p /home/user/.ssh; chmod 700 /home/user/.ssh; if [ -n \"$PUBLIC_KEY\" ]; then echo \"$PUBLIC_KEY\" >> /home/user/.ssh/authorized_keys; chmod 600 /home/user/.ssh/authorized_keys; chown -R user:user /home/user/.ssh; fi; sudo service ssh start; sleep infinity"]
