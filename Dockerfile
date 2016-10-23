FROM tensorflow/tensorflow:0.11.0rc1

MAINTAINER Nuno Silva <nuno.mgomes.silva@gmail.com>

# Update and install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	git-core \
	cmake \
	&& \ 
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Install python libraries
RUN pip install scikit-learn pyreadline

# Manage container shared folders
RUN rm -rf /shared/*
WORKDIR /shared
VOLUME [/shared]
CMD ["/run_jupyter.sh"]
