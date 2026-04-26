"""Launcher that fixes Host-header handling under NGINX upstream-vhost
rewrite, then invokes Chainlit's normal `chainlit run` CLI.

The problem: NGINX rewrites the Host header to chat-ui.chat.svc.cluster.local
(required for Istio's force-mtls DestinationRule routing). Chainlit's
OAuth error path calls request.url_for() which builds URLs from that
host — producing redirects to chat-ui.chat.svc.cluster.local that
the user's browser can't resolve.

Two fixes applied here, both before Chainlit starts:

  1. Monkeypatch uvicorn.Config to inject proxy_headers=True +
     forwarded_allow_ips='*'. This makes uvicorn substitute
     X-Forwarded-Proto into scope['scheme'], so request.url.scheme is
     'https' even though the upstream connection is plain HTTP.

  2. Add a Starlette-style ASGI middleware to Chainlit's app that
     reads X-Forwarded-Host (set by NGINX with the public hostname
     'chat.ekstest.com') and rewrites the 'host' header in the ASGI
     scope. uvicorn's proxy_headers does NOT do this — it only handles
     scheme + client IP. Without this middleware, request.url.netloc
     stays as the upstream Host and request.url_for() produces
     unresolvable URLs.

After both fixes, Chainlit's URL construction (via request.url_for or
get_user_facing_url) correctly produces public URLs.
"""
import sys

import uvicorn

# --- Fix 1: uvicorn.Config patch for proxy_headers --------------------------

_orig_config_init = uvicorn.Config.__init__


def _patched_config_init(self, *args, **kwargs):
    kwargs.setdefault("proxy_headers", True)
    kwargs.setdefault("forwarded_allow_ips", "*")
    return _orig_config_init(self, *args, **kwargs)


uvicorn.Config.__init__ = _patched_config_init


# --- Fix 2: ASGI middleware that rewrites Host from X-Forwarded-Host -------

class ForwardedHostMiddleware:
    """Rewrite the request's Host header from X-Forwarded-Host if present.

    NGINX's upstream-vhost annotation rewrites the Host header to the
    K8s Service DNS for Istio routing, but also sets X-Forwarded-Host
    to the original public hostname. Standard `proxy_headers` settings
    only handle client IP and scheme — not Host. This middleware closes
    that gap.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = list(scope.get("headers", []))
            forwarded_host = None
            for name, value in headers:
                if name.lower() == b"x-forwarded-host":
                    forwarded_host = value
                    break
            if forwarded_host:
                new_headers = [
                    (name, forwarded_host if name.lower() == b"host" else value)
                    for name, value in headers
                ]
                scope = {**scope, "headers": new_headers}
        await self.app(scope, receive, send)


# Mount the middleware on Chainlit's FastAPI app BEFORE the CLI starts the
# server. add_middleware appends to the user_middleware list which gets
# baked into the middleware stack on first request.
from chainlit.server import app as _chainlit_app

_chainlit_app.add_middleware(ForwardedHostMiddleware)


# --- Run Chainlit's normal CLI ---------------------------------------------

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
