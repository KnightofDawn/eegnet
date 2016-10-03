FROM tensorflow/tensorflow:0.10.0rc0

MAINTAINER Nuno Silva <nuno.mgomes.silva@gmail.com>

# Update and install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git-core \
	cmake

# Install OpenCV (big installation therefore separate)
RUN apt-get install -y --no-install-recommends \
	libopencv-dev \
	python-opencv

# Install OpenAI gym dependencies (https://github.com/openai/gym#id8)
RUN apt-get install -y --no-install-recommends \	
    zlib1g-dev \ 
    libjpeg-dev \ 
    xvfb \ 
    libav-tools \ 
    xorg-dev \ 
    python-opengl \ 
    libboost-all-dev \ 
    libsdl2-dev \ 
    swig

# Clean apt-get
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install OpenAI gym
RUN pip install gym[atari]

# Install python libraries
RUN pip install scikit-learn

# Manage container shared folders
RUN rm -rf /shared/*
WORKDIR /shared
VOLUME [/shared]
CMD ["/run_jupyter.sh"]
