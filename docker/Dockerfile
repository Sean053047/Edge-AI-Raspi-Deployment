FROM arm64v8/ubuntu:22.04

# * Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl bzip2 \
    ca-certificates \
    git tmux vim sudo \
    build-essential dkms linux-headers-raspi \
    software-properties-common \
    openssh-server openssh-client \
    dirmngr \
    gnupg2 \
    && add-apt-repository universe \
    && rm -rf /var/lib/apt/lists/*

# * Setup ssh server
RUN sed -i 's/#Port 22/Port 20000/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    service ssh start
EXPOSE 20000

# * Setup ROS2
RUN /bin/sh -c echo 'Etc/UTC' > /etc/timezone &&     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime &&     apt-get update &&     apt-get install -q -y --no-install-recommends tzdata &&     rm -rf /var/lib/apt/lists/* # buildkit
RUN /bin/sh -c set -eux;        key='C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654';        export GNUPGHOME="$(mktemp -d)";        gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key";        mkdir -p /usr/share/keyrings;        gpg --batch --export "$key" > /usr/share/keyrings/ros2-latest-archive-keyring.gpg;        gpgconf --kill all;        rm -rf "$GNUPGHOME" # buildkit
RUN /bin/sh -c echo "deb [ signed-by=/usr/share/keyrings/ros2-latest-archive-keyring.gpg ] http://packages.ros.org/ros2/ubuntu jammy main" > /etc/apt/sources.list.d/ros2-latest.list # buildkit
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get upgrade -y && \
    apt-get install -y ros-humble-ros-base ros-dev-tools ros-humble-vision-msgs ros-humble-cv-bridge && \
    rm -rf /var/lib/apt/lists/*


# * Set user
ARG UNAME=user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} -o ${UNAME}
RUN useradd -m -u ${UID} -g ${GID} -s /bin/bash ${UNAME}
RUN echo "${UNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# * Setup Hailo
COPY --chown=${UID}:${GID} hailo_setup /tmp/hailo_setup
RUN dpkg --unpack /tmp/hailo_setup/hailort_4.20.0_arm64.deb

# * Change USER
USER ${UNAME}
RUN mkdir /home/${UNAME}/.ssh && touch /home/${UNAME}/.ssh/authorized_keys 

CMD ["/bin/bash"]

# * Setup miniconda
COPY --chown=${UID}:${GID} requirements.txt /tmp/requirements.txt
# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /home/${UNAME}/miniconda.sh \
    && bash /home/${UNAME}/miniconda.sh -b -u -p /home/${UNAME}/miniconda3\
    && rm /home/${UNAME}/miniconda.sh
# Add conda to path
ENV PATH="/home/${UNAME}/miniconda3/bin:$PATH"
# Create conda environment
RUN conda create -yn edge-ai python=3.10 pip
# Set up shell to use conda environment by default
SHELL ["conda", "run", "-n", "edge-ai", "/bin/bash", "-c"]
RUN pip install -r /tmp/requirements.txt \
    && pip install /tmp/hailo_setup/hailort-4.20.0-cp310-cp310-linux_aarch64.whl

RUN conda init bash && \
    echo "conda activate edge-ai" >> ~/.bashrc && \ 
    echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
ENV PATH=/opt/conda/envs/edge-ai/bin:$PATH
ENV PYTHONPATH=/home/${UNAME}/miniconda3/envs/edge-ai/lib/python3.10/site-packages:$PYTHONPATH
WORKDIR /workspace
RUN sudo rosdep init && rosdep update

# RUN apt-get update && apt-get install -y \
#     libopencv-dev \
#     v4l-utils \
#     libv4l-dev \
#     ffmpeg \
#     uvcdynctrl \
    # && rm -rf /var/lib/apt/lists/*cl