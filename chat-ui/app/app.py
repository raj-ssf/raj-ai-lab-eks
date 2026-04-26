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
from typing import Any, Dict, Optional

import chainlit as cl
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("chat-ui")

# Chainlit 1.5+ ships Keycloak as a built-in OAuth provider — registered
# automatically when OAUTH_KEYCLOAK_CLIENT_ID, OAUTH_KEYCLOAK_CLIENT_SECRET,
# OAUTH_KEYCLOAK_BASE_URL, and OAUTH_KEYCLOAK_REALM are set in env. The
# framework wires up /auth/oauth/keycloak/login and .../callback at startup
# and uses Keycloak's standard OIDC endpoints. Earlier 1.3.x versions only
# had a runtime providers list (no route registration), which is why the
# previous custom-provider attempt looped.

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
async def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: Dict[str, str],
    default_user: cl.User,
    id_token: Optional[str] = None,
) -> Optional[cl.User]:
    """Stash the Keycloak access_token so on_message can forward it.

    Signature must match Chainlit 2.x's decorator contract exactly
    (chainlit/callbacks.py: oauth_callback expects an async function
    with five parameters returning Optional[User]). With a sync function
    or wrong arity, wrap_user_function silently fails and the resulting
    None user makes _authenticate_user raise 401 'credentialssignin'.
    The 5th parameter `id_token` is only populated for the Azure AD
    hybrid flow; for Keycloak it'll arrive as None — we accept it via
    a default so the same handler covers both.
    """
    log.info("oauth login: provider=%s user=%s", provider_id, default_user.identifier)
    # Stash the access_token in user.metadata, NOT cl.user_session.
    # cl.user_session is per-chat-session state, only available inside
    # @cl.on_chat_start / @cl.on_message handlers. During the OAuth
    # callback there's no chat session yet — calling
    # cl.user_session.set() here raises ChainlitContextException, the
    # handler returns None, _authenticate_user raises
    # 401 'credentialssignin' and the user lands at /login?error=...
    # user.metadata is serialized into the session JWT and accessible
    # later from any handler via cl.user_session.get("user").metadata.
    default_user.metadata["access_token"] = token
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

    # Retrieve the Keycloak access_token from the User object's metadata
    # (stashed there by @cl.oauth_callback). user_session.get("user")
    # returns the cl.User serialized into the session JWT cookie at
    # login time.
    user_obj = cl.user_session.get("user")
    token = user_obj.metadata.get("access_token") if user_obj else None
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
        # Chainlit 2.x renamed Action's fields: value → payload (dict),
        # description → tooltip. Old v1 names raise pydantic
        # ValidationError at construction, which aborts the
        # surrounding cl.Message.send() — the user sees Steps but
        # never the response bubble.
        actions.append(
            cl.Action(
                name="view_in_langfuse",
                payload={"trace_id": trace_id},
                label="🔍 View in Langfuse",
                tooltip="Open the matching trace in Langfuse",
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
    # action.payload is the dict we set on construction; read trace_id
    # back from it. (Chainlit 2.x removed the action.value attribute.)
    trace_id = action.payload.get("trace_id", "")
    url = f"{LANGFUSE_HOST}/trace/{trace_id}"
    await cl.Message(
        content=f"[Open trace in Langfuse]({url})",
    ).send()
