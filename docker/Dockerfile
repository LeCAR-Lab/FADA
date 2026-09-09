# Base image with IsaacSim 5.1
# Using IsaacSim as base for all environments (IsaacGym/MuJoCo/IsaacSim) to ensure
# compatibility across simulators
FROM nvcr.io/nvidia/isaac-sim:5.1.0

# Switch to root user for installation
USER root

# Environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV WORKSPACE_DIR=/workspace
ENV CONDA_ROOT=/root/.holosoma_deps/miniconda3
ENV PATH=$CONDA_ROOT/bin:$PATH

RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    swig \
    curl \
    wget \
    unzip \
    git \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    bash /miniconda.sh -b -u -p $CONDA_ROOT && \
    rm /miniconda.sh

# Configure conda for non-interactive use
RUN echo ". $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc && \
    conda config --set always_yes true

# Create workspace
RUN mkdir -p $WORKSPACE_DIR
WORKDIR $WORKSPACE_DIR

# Copy repository contents to holosoma
COPY . ./holosoma

##### Create wrapper script to handle setup scripts
##### Overrides sudo commands within those
RUN cat > /tmp/run_setup.sh << 'EOF'
#!/bin/bash
set -e

# Override sudo since it doesn't work in docker build
function sudo() { "$@"; }
export -f sudo

# Set up conda environment
source $CONDA_ROOT/etc/profile.d/conda.sh

cd /workspace/holosoma/scripts
chmod +x setup_isaacsim.sh setup_isaacgym.sh setup_mujoco.sh setup_inference.sh setup_retargeting.sh
./setup_isaacsim.sh
./setup_isaacgym.sh
./setup_mujoco.sh --no-warp
./setup_inference.sh
./setup_retargeting.sh
EOF

##### Make wrapper executable and run setup
RUN chmod +x /tmp/run_setup.sh && \
    bash /tmp/run_setup.sh

# Set up conda activation for runtime
RUN echo "source $CONDA_ROOT/etc/profile.d/conda.sh" >> ~/.bashrc

# Set working directory
WORKDIR /workspace/holosoma

ENTRYPOINT []
CMD ["/bin/bash"]
