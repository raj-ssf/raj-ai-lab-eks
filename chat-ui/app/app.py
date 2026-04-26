"""chat-ui — Chainlit-based UI for the Raj AI Lab's LangGraph router.

Two flows:

  - chat: types a prompt → POST /invoke on langgraph-service → render
    each StateGraph node (classify / ensure_warm / execute) as a
    `cl.Step` with measured timing, plus the response message with a
    JSON sidebar and a deep-link to the matching Langfuse trace.

  - upload (placeholder): /upload command opens a file picker and
    acknowledges the upload. Real ingestion lands in the next iteration.

Auth: Chainlit's `@cl.oauth_callback` runs Authorization Code + PKCE
against Keycloak (CONFIDENTIAL client `chat-ui` configured via
OAUTH_KEYCLOAK_* env vars from chat-ui-oidc Secret). The access_token
is forwarded to langgraph-service /invoke as the Bearer token.
"""

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import chainlit as cl
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("chat-ui")


# -- Custom Keycloak OAuth provider ------------------------------------------
#
# Chainlit 1.3.x ships Google / GitHub / Azure AD / Okta / Auth0 / Descope /
# AWS Cognito / GitLab as built-in providers but NOT Keycloak (added in a
# later release). To use Keycloak without bumping Chainlit, we subclass
# OAuthProvider, configure it from the OAUTH_KEYCLOAK_* env vars set in the
# Deployment, and register it BEFORE @cl.oauth_callback is evaluated below.
#
# The framework calls get_token() with the auth-code returned from
# Keycloak's redirect, and get_user_info() with the resulting access token.
# We return Keycloak's `preferred_username` (the realm-defined human
# username) as the cl.User.identifier — same value that flows through to
# JWT.preferred_username on calls to langgraph-service /invoke.
from chainlit.oauth_providers import OAuthProvider, providers


class KeycloakOAuthProvider(OAuthProvider):
    id = "keycloak"
    env = [
        "OAUTH_KEYCLOAK_CLIENT_ID",
        "OAUTH_KEYCLOAK_CLIENT_SECRET",
        "OAUTH_KEYCLOAK_BASE_URL",
        "OAUTH_KEYCLOAK_REALM",
    ]

    def __init__(self) -> None:
        self.client_id = os.environ["OAUTH_KEYCLOAK_CLIENT_ID"]
        self.client_secret = os.environ["OAUTH_KEYCLOAK_CLIENT_SECRET"]
        base = os.environ["OAUTH_KEYCLOAK_BASE_URL"].rstrip("/")
        realm = os.environ["OAUTH_KEYCLOAK_REALM"]
        # Display name on Chainlit's "Sign in with X" button. Falls back
        # to a sensible default if the env var isn't set.
        self.name = os.environ.get("OAUTH_KEYCLOAK_NAME", "Keycloak")
        # Standard Keycloak OIDC endpoints. The realm ID and base URL
        # together fully determine these — no Keycloak admin API access
        # needed at runtime.
        self.authorize_url = f"{base}/realms/{realm}/protocol/openid-connect/auth"
        self.token_url = f"{base}/realms/{realm}/protocol/openid-connect/token"
        self.userinfo_url = f"{base}/realms/{realm}/protocol/openid-connect/userinfo"
        # Additional auth-request params. `scope` declares which OIDC
        # claims we want; `openid` is required, `profile` and `email`
        # populate preferred_username / email on the userinfo response.
        self.authorize_params = {
            "scope": "openid profile email",
        }

    async def get_token(self, code: str, url: str) -> str:
        """Exchange the auth code for an access token.

        Called by Chainlit on the /auth/oauth/keycloak/callback handler
        after Keycloak redirects back with ?code=...&state=... .
        """
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": url,
                },
            )
            r.raise_for_status()
            return r.json()["access_token"]

    async def get_user_info(self, token: str) -> Tuple[Dict[str, Any], cl.User]:
        """Fetch user claims from Keycloak's userinfo endpoint.

        Returns the raw claim dict (handed back to oauth_callback as
        raw_user_data) plus a cl.User keyed on preferred_username.
        """
        async with httpx.AsyncClient(verify=False) as client:
            r = await client.get(
                self.userinfo_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            r.raise_for_status()
            data = r.json()
        identifier = data.get("preferred_username") or data.get("sub") or "unknown"
        user = cl.User(
            identifier=identifier,
            metadata={"provider": "keycloak", "email": data.get("email", "")},
        )
        return data, user


# Register our provider before @cl.oauth_callback runs at import time.
# Chainlit's "is any provider configured?" check iterates this list; with
# our entry present and env vars set, the check passes and the framework
# wires up routes /auth/oauth/keycloak/login and
# /auth/oauth/keycloak/callback automatically.
providers.append(KeycloakOAuthProvider())

# In-cluster URL — direct mesh path, not the public ingress. The mesh's
# ISTIO_MUTUAL DestinationRule + the allow-chat-ui AuthorizationPolicy
# in the langgraph namespace let this connection through.
LANGGRAPH_URL = os.environ.get(
    "LANGGRAPH_URL", "http://langgraph-service.langgraph.svc.cluster.local"
)

# Public Langfuse host for "view trace" deep-links. The link goes to a
# user's browser, not the pod — so it has to be the public hostname,
# not the in-cluster Service URL.
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://langfuse.ekstest.com")


# -- Auth --------------------------------------------------------------------

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    """Stash the Keycloak access_token so on_message can forward it.

    Chainlit's User object carries an `identifier` (sub claim) and any
    metadata we want; we add the token to user_session so it's available
    in subsequent handlers without ever ending up in the User object's
    serialized form.
    """
    cl.user_session.set("access_token", token)
    log.info("oauth login: provider=%s user=%s", provider_id, default_user.identifier)
    return default_user


# -- Welcome -----------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    user = cl.user_session.get("user")
    name = user.identifier if user else "guest"
    await cl.Message(
        content=(
            f"Hi **{name}** — type a prompt and the LangGraph router will "
            f"classify, route, and execute it. Use `/upload` to upload a "
            f"document (ingestion pipeline placeholder for now)."
        ),
    ).send()


# -- Upload placeholder ------------------------------------------------------

async def _handle_upload() -> None:
    files = await cl.AskFileMessage(
        content="Drop a PDF, DOCX, or TXT (max 10 MB).",
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
        ],
        max_size_mb=10,
        timeout=300,
    ).send()
    if not files:
        return
    f = files[0]
    await cl.Message(
        content=(
            f"Received **{f.name}** ({f.size:,} bytes, {f.type}). "
            f"The ingestion service ships in the next iteration; the file is "
            f"acknowledged but not stored or processed yet."
        ),
    ).send()


# -- Chat handler ------------------------------------------------------------

async def _call_invoke(prompt: str, token: str) -> Dict[str, Any]:
    """Single synchronous POST to langgraph-service /invoke."""
    async with httpx.AsyncClient(verify=False, timeout=600.0) as client:
        r = await client.post(
            f"{LANGGRAPH_URL}/invoke",
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
            },
            json={"prompt": prompt, "max_tokens": 600},
        )
        r.raise_for_status()
        return r.json()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    if message.content.strip().lower() in ("/upload", "upload"):
        await _handle_upload()
        return

    token = cl.user_session.get("access_token")
    if not token:
        await cl.Message(content="Not authenticated. Refresh and log in again.").send()
        return

    prompt = message.content
    t0 = time.time()

    # Outer step wraps the entire /invoke. Inside, we render the three
    # state-graph nodes as nested steps with their measured timings
    # AFTER the response returns (the call is synchronous, not streamed,
    # so we render retroactively rather than in real time).
    async with cl.Step(name="LangGraph router", type="run") as outer:
        outer.input = prompt
        try:
            data = await _call_invoke(prompt, token)
        except httpx.HTTPStatusError as e:
            outer.is_error = True
            outer.output = f"upstream returned {e.response.status_code}: {e.response.text[:200]}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return
        except Exception as e:
            outer.is_error = True
            outer.output = f"{type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return

        outer.output = (
            f"route={data['route']}  "
            f"cold_start={data['cold_start']}  "
            f"latency={data['execute_latency_ms']}ms  "
            f"total={int((time.time() - t0) * 1000)}ms"
        )

        # Per-node step rendering. The actual timings come from
        # langgraph-service's response (warm_wait_seconds,
        # execute_latency_ms); we don't have a per-node breakdown for
        # classify, so we elapsed-time-bound it loosely.
        async with cl.Step(name="classify", type="llm", parent_id=outer.id) as s:
            s.input = "Categorize the prompt: trivial | reasoning | hard"
            classifier_raw = (data.get("classifier_raw") or "").strip()
            preview = classifier_raw if len(classifier_raw) <= 200 else classifier_raw[:200] + "…"
            s.output = (
                f"**route**: `{data['route']}`\n\n"
                f"**raw classifier output**: `{preview}`"
            )

        async with cl.Step(name="ensure_warm", type="tool", parent_id=outer.id) as s:
            if data.get("cold_start"):
                s.output = (
                    f"❄️ **cold start** — Karpenter scaled up the target "
                    f"GPU node and loaded the model in "
                    f"**{data['warm_wait_seconds']:.1f}s**."
                )
            else:
                s.output = "✅ already warm — scale call was a no-op."

        async with cl.Step(name="execute", type="llm", parent_id=outer.id) as s:
            s.input = prompt
            s.output = (
                f"**latency**: {data['execute_latency_ms']} ms  "
                f"(model selected by router for route '{data['route']}')"
            )

    # Final response bubble. JSON sidebar shows the full /invoke shape;
    # actions row gives a one-click jump to the matching Langfuse trace.
    elements = [
        cl.Text(
            name="routing.json",
            content=json.dumps(data, indent=2),
            display="side",
        ),
    ]
    actions = []
    trace_id = data.get("langfuse_trace_id")
    if trace_id:
        actions.append(
            cl.Action(
                name="view_in_langfuse",
                value=trace_id,
                description="Open the matching trace in Langfuse",
                label="🔍 View in Langfuse",
            )
        )

    await cl.Message(
        content=data.get("response", ""),
        elements=elements,
        actions=actions,
    ).send()


@cl.action_callback("view_in_langfuse")
async def on_view_trace(action: cl.Action) -> None:
    """Open the Langfuse trace UI in a new tab."""
    url = f"{LANGFUSE_HOST}/trace/{action.value}"
    await cl.Message(
        content=f"[Open trace in Langfuse]({url})",
    ).send()
