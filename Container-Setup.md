# Container
We are using an uv python container for faster build times using the uv package manager.

# Setting up
If on windows open the docker desktop app.

Run:

`docker build -t lucas/changepoint-app .`

`docker run --volume .:/app --publish 8050:8000 lucas/changepoint-app`

`docker run --volume .:/app --volume ./tmp-data:/tmp-data --publish 8050:8000 lucas/changepoint-app`
`podman run -d --volume ./:/app:z --volume ../tmp-data:/tmp-data-folder:z --publish 8443:8000 lucas/changepoint-app`