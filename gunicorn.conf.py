# gunicorn.conf.py
import os
import ssl

bind = "0.0.0.0:8000"
certfile = "Cert.pem"
keyfile = "privateKey.pem"
timeout = 300

# optional
workers = 4
threads = 4

# load the data on worker restart
preload_app = True


def ssl_context(conf, default_ssl_context_factory):
    password = os.environ["SSL_KEY_PASSWORD"]   # set this in the environment

    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.load_cert_chain(
        certfile=conf.certfile,
        keyfile=conf.keyfile,
        password=password,   # can also be a callback function
    )

    # optional hardening / tuning
    # ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx
