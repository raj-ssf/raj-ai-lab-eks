"""Launcher that monkeypatches uvicorn.Config to enable proxy-headers
parsing, then invokes Chainlit's normal `chainlit run` CLI.

Why this exists: Chainlit's CLI constructs uvicorn.Config without
proxy_headers=True, so Starlette's request.url (and therefore
request.url_for() inside Chainlit's OAuth error path) reads the
upstream Host header — which NGINX rewrites to
chat-ui.chat.svc.cluster.local for Istio mTLS routing. Result: any
OAuth error redirect Location ends up pointing to the in-cluster
Service URL, browser fails DNS resolution.

Patching uvicorn.Config to inject proxy_headers=True +
forwarded_allow_ips='*' makes uvicorn replace request.url's host/scheme
with X-Forwarded-Host/X-Forwarded-Proto (set correctly by NGINX with
the public hostname chat.ekstest.com) before any handler sees them.
"""
import sys

import uvicorn

_orig_config_init = uvicorn.Config.__init__


def _patched_config_init(self, *args, **kwargs):
    kwargs.setdefault("proxy_headers", True)
    kwargs.setdefault("forwarded_allow_ips", "*")
    return _orig_config_init(self, *args, **kwargs)


uvicorn.Config.__init__ = _patched_config_init

# Forward to chainlit's normal CLI. argv[0] is the CLI's name, so we
# swap it to "chainlit" and pass the rest of the run command.
sys.argv = [
    "chainlit",
    "run",
    "app.py",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "-h",
]

from chainlit.cli import cli

cli()
