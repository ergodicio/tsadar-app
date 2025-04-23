# base image
FROM mambaorg/micromamba:latest

WORKDIR /runner

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/runner/env_gpu.yml docker/runner/fit_for_app.py docker/runner/run.sh /runner/
RUN chmod +x run.sh

RUN micromamba install -y -n base -f /runner/env_gpu.yml && \
    micromamba clean --all --yes
