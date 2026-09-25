FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        doxygen \
        graphviz \
    && rm -rf /var/lib/apt/lists/*
