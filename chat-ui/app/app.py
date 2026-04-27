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

import asyncio
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

# Phase 5: in-cluster ingestion-service URL. allow-chat-ui AuthZ in the
# ingestion namespace (ingestion-service.tf) lets the chat-ui SA POST
# /upload + GET /jobs/{id}.
INGESTION_URL = os.environ.get(
    "INGESTION_URL", "http://ingestion-service.ingestion.svc.cluster.local"
)

# Public Langfuse host for "view trace" deep-links. The link goes to a
# user's browser, not the pod — so it has to be the public hostname,
# not the in-cluster Service URL.
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://langfuse.ekstest.com")

# Polling cadence for the upload job state machine. Ingestion runs as
# a FastAPI BackgroundTask so /upload returns 202 immediately; we
# follow up with GET /jobs/{id} until state hits "done" or "failed".
# Sub-second cadence keeps the UI feeling live without hammering the
# server — the typical end-to-end is 5-30s for a small doc.
UPLOAD_POLL_INTERVAL_SECONDS = 1.0
UPLOAD_POLL_TIMEOUT_SECONDS = 600.0


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


# -- Helpers -----------------------------------------------------------------

def _thread_id() -> str:
    """Stable conversation identifier used as session_id across flows.

    Chainlit 2.x exposes `cl.context.session.thread_id` — a UUID assigned
    when the WebSocket session opens and persisted across reconnects (when
    a data layer is configured) or for the lifetime of the connection
    (when not). Chat-ui has no data layer today; the thread is per-tab.

    Both /upload and /invoke pass this string as session_id so a doc
    uploaded in this conversation surfaces to the retrieve node when the
    user asks a question, but doesn't leak across browser tabs / threads.
    """
    return cl.context.session.thread_id


def _user_token() -> Optional[str]:
    """Pull the Keycloak access_token stashed in user.metadata at login.

    Returns None if the user object is missing (shouldn't happen post-OAuth)
    or the token isn't present (defensive — also shouldn't happen)."""
    user_obj = cl.user_session.get("user")
    return user_obj.metadata.get("access_token") if user_obj else None


# -- Welcome -----------------------------------------------------------------

@cl.on_chat_start
async def on_chat_start() -> None:
    user = cl.user_session.get("user")
    name = user.identifier if user else "guest"
    await cl.Message(
        content=(
            f"Hi **{name}** — type a prompt and the LangGraph router will "
            f"classify, route, and execute it. Use `/upload` to add a document "
            f"to **this conversation's** corpus; subsequent questions will be "
            f"grounded in any uploaded text via per-thread RAG."
        ),
    ).send()


# -- Upload (real) -----------------------------------------------------------
#
# Flow per /upload command:
#   1. cl.AskFileMessage opens the file picker.
#   2. POST multipart/form-data to ingestion-service /upload with the
#      user's bearer token + thread_id as session_id; expect 202 + job_id.
#   3. Poll GET /jobs/{job_id} once a second; render each new state as a
#      nested cl.Step under the outer "Ingestion" step.
#   4. On done: report chunk count + tell the user they can now ask
#      questions about the doc. On failed: surface the detail string from
#      the job record.
#
# Uses BackgroundTasks server-side, so /upload returns immediately and
# the client polls. The FastAPI handler holds JOBS in-memory (single pod,
# no persistence — fine for our scale).

# Map the ingestion-service state machine values to (label, emoji) tuples
# for the cl.Step output. States in order: received → parsing → chunking
# → embedding → writing → done. "failed" is terminal at any point.
_UPLOAD_STATE_LABELS = {
    "received":  ("📥 Received",        "uploaded to ingestion-service"),
    "parsing":   ("📄 Parsing",         "Unstructured.partition() reading the file"),
    "chunking":  ("✂️  Chunking",        "RecursiveCharacterTextSplitter (1000/200)"),
    "embedding": ("🧮 Embedding",       "bge-m3 batch embeddings via vllm"),
    "writing":   ("💾 Writing",         "upserting points into Qdrant"),
    "done":      ("✅ Done",            "ingestion complete"),
    "failed":    ("❌ Failed",          "see detail"),
}


async def _post_upload(
    file: cl.AskFileResponse, token: str, session_id: str
) -> Dict[str, Any]:
    """POST multipart/form-data to ingestion-service /upload. Returns the
    202 body with job_id."""
    # cl.AskFileResponse stores bytes at .path on disk. Read them in;
    # ingestion-service has a 25 MiB cap that mirrors the chat-ui cap so
    # we shouldn't see large files here.
    with open(file.path, "rb") as fp:
        data = fp.read()
    async with httpx.AsyncClient(verify=False, timeout=120.0) as client:
        r = await client.post(
            f"{INGESTION_URL}/upload",
            headers={"authorization": f"Bearer {token}"},
            files={"file": (file.name, data, file.type)},
            data={"session_id": session_id},
        )
        r.raise_for_status()
        return r.json()


async def _poll_job(job_id: str, token: str) -> Dict[str, Any]:
    """Poll /jobs/{job_id} until terminal state or timeout. Returns the
    final job record."""
    deadline = time.monotonic() + UPLOAD_POLL_TIMEOUT_SECONDS
    async with httpx.AsyncClient(verify=False, timeout=10.0) as client:
        while time.monotonic() < deadline:
            r = await client.get(
                f"{INGESTION_URL}/jobs/{job_id}",
                headers={"authorization": f"Bearer {token}"},
            )
            r.raise_for_status()
            job = r.json()
            if job.get("state") in ("done", "failed"):
                return job
            await asyncio.sleep(UPLOAD_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"ingestion job {job_id} did not finish within "
        f"{UPLOAD_POLL_TIMEOUT_SECONDS:.0f}s"
    )


async def _handle_upload() -> None:
    """End-to-end /upload command: pick file → POST → poll → report."""
    token = _user_token()
    if not token:
        await cl.Message(content="Not authenticated. Refresh and log in again.").send()
        return

    files = await cl.AskFileMessage(
        content=(
            "Drop a PDF, DOCX, or TXT (max 10 MB). "
            "The doc will be parsed, chunked, embedded with bge-m3, and "
            "stored in Qdrant under this conversation's session_id."
        ),
        accept=[
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
        ],
        max_size_mb=10,
        timeout=300,
    ).send()
    if not files:
        return
    f = files[0]
    session_id = _thread_id()

    # Outer step wraps the whole pipeline. Each state transition lands as
    # a child step; when done/failed, the outer's output summarizes.
    async with cl.Step(name="Ingestion", type="run") as outer:
        outer.input = (
            f"file={f.name}  bytes={f.size:,}  type={f.type}  "
            f"session_id={session_id[:8]}…"
        )

        # Submit
        try:
            job = await _post_upload(f, token, session_id)
        except httpx.HTTPStatusError as e:
            outer.is_error = True
            outer.output = f"upload submit failed: {e.response.status_code} {e.response.text[:200]}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return
        except Exception as e:
            outer.is_error = True
            outer.output = f"upload submit failed: {type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return

        job_id = job["job_id"]

        # Poll, rendering each new state as a child step. We track which
        # states we've already shown so we don't duplicate when the
        # server stays in one state across multiple poll ticks.
        shown: set[str] = set()
        # Initial state ("received") is always present in the 202 body
        # — render it before polling kicks in so the user sees activity.
        async def _render_state(state: str, detail: Optional[str]) -> None:
            label, hint = _UPLOAD_STATE_LABELS.get(state, (state, ""))
            async with cl.Step(name=label, type="tool", parent_id=outer.id) as s:
                if state == "failed":
                    s.is_error = True
                    s.output = detail or hint
                else:
                    s.output = hint

        await _render_state(job.get("state", "received"), job.get("detail"))
        shown.add(job.get("state", "received"))

        try:
            final = await _poll_job(job_id, token)
        except Exception as e:
            outer.is_error = True
            outer.output = f"polling failed: {type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return

        # Replay the full state sequence. We didn't watch every transition
        # (a 1s poll can skip rapid intermediate states), but the canonical
        # ordering is fixed — render any missing intermediate state for a
        # readable trail. Skip "received" (already shown) and only emit
        # states we haven't already.
        for state in ("parsing", "chunking", "embedding", "writing", final["state"]):
            if state in shown:
                continue
            await _render_state(
                state,
                final.get("detail") if state == final["state"] else None,
            )
            shown.add(state)

        if final["state"] == "done":
            chunks = final.get("chunks_written", 0)
            outer.output = (
                f"ingested **{f.name}** → {chunks} chunk(s) in Qdrant "
                f"(session_id={session_id[:8]}…)"
            )
            await cl.Message(
                content=(
                    f"✅ **{f.name}** ingested — {chunks} chunk(s) embedded into "
                    f"this conversation's corpus. Ask a question about it."
                ),
            ).send()
        else:
            outer.is_error = True
            outer.output = f"ingestion failed: {final.get('detail') or 'unknown'}"
            await cl.Message(
                content=(
                    f"❌ Ingestion of **{f.name}** failed: "
                    f"`{final.get('detail') or 'unknown'}`"
                ),
            ).send()


# -- Chat handler ------------------------------------------------------------

async def _call_invoke(prompt: str, token: str, session_id: str) -> Dict[str, Any]:
    """Single synchronous POST to langgraph-service /invoke.

    Phase 5: forwards session_id (Chainlit thread_id) so the LangGraph
    retrieve node can pull this conversation's uploaded chunks. When no
    documents have been uploaded yet to this thread, the retrieve node
    short-circuits to zero hits and the prompt runs unaugmented — same
    behavior as before Phase 5.
    """
    async with httpx.AsyncClient(verify=False, timeout=600.0) as client:
        r = await client.post(
            f"{LANGGRAPH_URL}/invoke",
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
            },
            json={
                "prompt":     prompt,
                "max_tokens": 600,
                "session_id": session_id,
                "top_k":      5,
            },
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
    token = _user_token()
    if not token:
        await cl.Message(content="Not authenticated. Refresh and log in again.").send()
        return

    prompt = message.content
    session_id = _thread_id()
    t0 = time.time()

    # Outer step wraps the entire /invoke. Inside, we render the four
    # state-graph nodes as nested steps with their measured timings
    # AFTER the response returns (the call is synchronous, not streamed,
    # so we render retroactively rather than in real time).
    async with cl.Step(name="LangGraph router", type="run") as outer:
        outer.input = prompt
        try:
            data = await _call_invoke(prompt, token, session_id)
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

        retrieve_count = data.get("retrieve_count", 0)
        outer.output = (
            f"route={data['route']}  "
            f"retrieved={retrieve_count}  "
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

        # Phase 5: render the retrieve step. retrieve_count==0 is the
        # common case (nothing uploaded to this thread yet) and isn't
        # an error — we just say so.
        async with cl.Step(name="retrieve", type="retrieval", parent_id=outer.id) as s:
            s.input = f"query={prompt[:100]}  session_id={session_id[:8]}…"
            retrieve_ms = data.get("retrieve_ms", 0)
            if retrieve_count > 0:
                # Show a compact preview of each chunk in the step output.
                # The full text lives in the side-panel "sources.md" element
                # rendered after the run.
                lines = []
                for i, c in enumerate(data.get("retrieved_chunks", []), start=1):
                    src = c.get("source", "?")
                    score = c.get("score", 0.0)
                    preview = (c.get("text") or "")[:120].replace("\n", " ")
                    lines.append(f"**{i}.** `{src}` (score={score:.3f}) — {preview}…")
                s.output = (
                    f"📚 **{retrieve_count} chunk(s)** retrieved in **{retrieve_ms} ms**\n\n"
                    + "\n\n".join(lines)
                )
            else:
                s.output = (
                    f"no documents in this conversation's corpus "
                    f"(retrieve_ms={retrieve_ms}). Use `/upload` to add one."
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
    # Phase 5: when chunks were retrieved, attach a Sources side panel
    # so the user can audit what context grounded the answer.
    if retrieve_count > 0:
        sources_md_lines = ["# Sources used as RAG context", ""]
        for i, c in enumerate(data.get("retrieved_chunks", []), start=1):
            src = c.get("source", "?")
            chunk_idx = c.get("chunk_index", 0)
            score = c.get("score", 0.0)
            text = c.get("text", "")
            sources_md_lines.append(
                f"## [{i}] `{src}` — chunk {chunk_idx} (score={score:.3f})\n\n{text}\n"
            )
        elements.append(
            cl.Text(
                name="sources.md",
                content="\n".join(sources_md_lines),
                display="side",
            )
        )

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
