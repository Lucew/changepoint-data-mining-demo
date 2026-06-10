# Container
We are using an uv python container for faster build times using the uv package manager.

# Setting up
If on windows open the docker desktop app.

1) Run to build the container:

`docker build -t lucas/changepoint-app .`

2) Set the password before running the container:

`export SSL_KEY_PASSWORD='your-key-password'`

3) Run the container:

`docker run --env SSL_KEY_PASSWORD='your-password' --volume .:/app --volume ./tmp-data:/tmp-data --publish 8050:8000 lucas/changepoint-app`

`podman run -d --env SSL_KEY_PASSWORD='your-password' --volume ./:/app:z --volume ../tmp-data:/tmp-data-folder:z --publish 8443:8000 lucas/changepoint-app`

# Enable restart

1. Find the container name

`podman ps -a`

3. Set boot restart behavior on the existing container

`podman update --restart=unless-stopped <container-name>`

or:

`podman update --restart=always <container-name>`

3. Enable Podman's restart service for your user

`systemctl --user enable podman-restart.service`

4. Make your user systemd instance start at boot, even before login

`sudo loginctl enable-linger "$USER"`
