# base image
FROM mambaorg/micromamba:latest

WORKDIR /app

COPY --chown=$MAMBA_USER:$MAMBA_USER tsadar_gui /app/tsadar_gui
COPY --chown=$MAMBA_USER:$MAMBA_USER mambaenv.yaml tsadar_app.py requirements.txt /app/

RUN micromamba install -y -n base -f /app/mambaenv.yaml && \
    micromamba clean --all --yes

EXPOSE 8501

CMD micromamba run -n base streamlit run /app/tsadar_app.py \
    --server.headless=true \
    --browser.serverAddress="0.0.0.0" \
    --server.port=8501 \
    --browser.gatherUsageStats=false \
    --server.baseUrlPath="/tsadar"