FROM astral/uv:python3.12-bookworm-slim
LABEL authors="lucas"

# get the requirements
COPY requirements.txt .

# install the requirements
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system gunicorn

# make an app directory
WORKDIR /app

ENTRYPOINT ["gunicorn", "Dash_Mainpage:server", "--workers", "1", "--bind", "0.0.0.0:8000", "--timeout", "300", "--certfile", "Cert.pem", "--keyfile", "privateKey.pem"]