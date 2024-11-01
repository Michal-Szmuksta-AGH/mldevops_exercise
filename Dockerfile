# Wybierz obraz bazowy z CUDA i PyTorch
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime AS base

# Instalacja zależności systemowych
RUN apt update && apt install -y \
        build-essential \
        default-libmysqlclient-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
        pkg-config \
        python3-dev \
        curl \
    && apt clean

# Kopiowanie i instalacja pakietu autoencoders
COPY ./dist/autoencoders-*.whl .
RUN pip install autoencoders-*.whl

# Tworzenie użytkownika
ENV UID=1001 GID=1001
RUN addgroup --gid ${GID} --system user && \
    adduser --shell /bin/bash --disabled-password --uid ${UID} --system --group user

# Konfiguracja JupyterLab
EXPOSE 8888
WORKDIR /home/user/autoencoders
USER user
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]