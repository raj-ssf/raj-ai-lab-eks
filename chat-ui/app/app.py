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
from prometheus_client import Counter, Histogram, start_http_server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("chat-ui")

# Phase #47: Prometheus instrumentation on a side port.
#
# Chainlit doesn't expose its internal FastAPI app for clean
# prometheus_fastapi_instrumentator hookup the way ingestion-service
# / rag-service / langgraph-service do. Instead we run a separate
# HTTP server on port 8001 (just /metrics) using prometheus_client's
# start_http_server(). The chat-ui Service exposes both ports —
# 8000 for the Chainlit UI/API + 8001 for /metrics — and a
# ServiceMonitor scrapes 8001.
#
# Manual instrumentation only — we don't get auto HTTP-level series
# like the FastAPI Instrumentator. The counters/histograms below
# are wrapped around the call sites that matter (langgraph /invoke,
# ingestion /upload) so canary AnalysisTemplates can later gate on
# them in a future Phase #47b.
CHAT_INVOKE_TOTAL = Counter(
    "chat_invoke_total",
    "Total /invoke calls from chat-ui to langgraph-service.",
    ["outcome"],  # success | error
)
CHAT_INVOKE_DURATION_SECONDS = Histogram(
    "chat_invoke_duration_seconds",
    "End-to-end /invoke latency from chat-ui's perspective.",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0),
)
CHAT_UPLOAD_TOTAL = Counter(
    "chat_upload_total",
    "Total /upload calls from chat-ui to ingestion-service.",
    ["outcome"],  # success | error
)
CHAT_OAUTH_TOTAL = Counter(
    "chat_oauth_total",
    "Total OAuth callback events.",
    ["outcome"],  # accepted | rejected
)

# Start the /metrics HTTP server. Port 8001, all interfaces. Idempotent
# since chat-ui runs single-replica; multi-replica would still be fine
# because each pod listens on its own pod IP.
_METRICS_PORT = int(os.environ.get("METRICS_PORT", "8001"))
try:
    start_http_server(_METRICS_PORT)
    log.info("prometheus /metrics server listening on :%d", _METRICS_PORT)
except OSError as e:
    # Port-in-use can happen on local hot-reload during dev; in-cluster
    # the pod is fresh on every restart so this is purely defensive.
    log.warning("metrics server failed to start: %s", e)

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

# Phase #21: SSE streaming via /invoke/stream. When true, chat-ui
# consumes Server-Sent Events from langgraph-service and renders
# tokens + node-progress events progressively. When false, falls
# back to the legacy synchronous /invoke + retroactive cl.Step
# rendering — useful for debugging or when SSE through some
# intermediate proxy proves unreliable.
STREAM_ENABLED = os.environ.get("STREAM_ENABLED", "true").lower() in (
    "1", "true", "yes",
)

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
    file: Any, token: str, session_id: str
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


# -- /export + /delete-session (Phase #22) -----------------------------------
#
# UX-side companion to Phase #17's GDPR primitives. Two slash commands:
#   /export           Fetch /session/<id>/export + /feedback/stats from
#                     langgraph-service, format as a downloadable
#                     markdown file via cl.File.
#   /delete-session   Call DELETE /session/<id> on langgraph-service to
#                     atomically wipe per-(user, session) data — memory
#                     turns/summary + cache entries. Does NOT clear
#                     conversation history rendered in the Chainlit
#                     thread itself (that's the user's local UI state).


async def _call_session_export(
    token: str, session_id: str
) -> Dict[str, Any]:
    """GET /session/{session_id}/export. Raises on HTTP error."""
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        r = await client.get(
            f"{LANGGRAPH_URL}/session/{session_id}/export",
            headers={"authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


async def _call_feedback_stats(token: str, limit: int = 50) -> Dict[str, Any]:
    """GET /feedback/stats?limit=N. Returns {} on error (best-effort —
    feedback is enrichment for the export, not the critical path)."""
    try:
        async with httpx.AsyncClient(verify=False, timeout=15.0) as client:
            r = await client.get(
                f"{LANGGRAPH_URL}/feedback/stats?limit={limit}",
                headers={"authorization": f"Bearer {token}"},
            )
            r.raise_for_status()
            return r.json()
    except Exception:
        return {}


async def _call_session_delete(
    token: str, session_id: str
) -> Dict[str, Any]:
    """DELETE /session/{session_id}. Raises on HTTP error."""
    async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
        r = await client.delete(
            f"{LANGGRAPH_URL}/session/{session_id}",
            headers={"authorization": f"Bearer {token}"},
        )
        r.raise_for_status()
        return r.json()


def _format_export_markdown(
    export: Dict[str, Any], stats: Dict[str, Any]
) -> str:
    """Render the export payload as a single human-readable markdown
    document. Sections: header, memory summary, recent turns, cache
    entries, recent feedback. Designed to be useful both as a
    download (auditor reads it) and as a paste-into-chat reference
    if the user wants to resume a conversation later."""
    from datetime import datetime, timezone
    import json as _json

    user = export.get("user", "unknown")
    session_id = export.get("session_id", "unknown")
    captured_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# Session export")
    lines.append("")
    lines.append(f"- **User**: `{user}`")
    lines.append(f"- **Session ID**: `{session_id}`")
    lines.append(f"- **Captured at**: {captured_at}")
    lines.append(
        f"- **Memory turns**: {export.get('memory_turn_count', 0)}"
    )
    lines.append(
        f"- **Cache entries**: {export.get('cache_entry_count', 0)}"
    )
    if stats:
        lines.append(
            f"- **Feedback (all sessions)**: total={stats.get('total', 0)} "
            f"up={stats.get('up', 0)} down={stats.get('down', 0)}"
        )
    lines.append("")

    # Long-term summary
    summary = export.get("memory_summary") or ""
    lines.append("## Conversation summary")
    lines.append("")
    if summary:
        lines.append(summary)
    else:
        lines.append("_(none yet — summary is regenerated every "
                     "MEMORY_SUMMARIZE_AFTER_TURNS turns)_")
    lines.append("")

    # Recent turns
    lines.append("## Recent turns")
    lines.append("")
    turns = export.get("memory_turns") or []
    if not turns:
        lines.append("_(no turns saved yet)_")
    else:
        # Memory list is newest-first (LPUSH); render chronologically
        for turn in reversed(turns):
            ts_raw = turn.get("ts", 0)
            try:
                ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).isoformat()
            except (TypeError, ValueError):
                ts = "unknown"
            lines.append(f"### {ts}")
            lines.append("")
            lines.append("**User:**")
            lines.append("")
            lines.append("```")
            lines.append(turn.get("prompt", ""))
            lines.append("```")
            lines.append("")
            lines.append("**Assistant:**")
            lines.append("")
            lines.append(turn.get("response", ""))
            lines.append("")
    lines.append("")

    # Cache entries
    lines.append("## Cache entries")
    lines.append("")
    cache_entries = export.get("cache_entries") or []
    if not cache_entries:
        lines.append("_(no cached responses for this session)_")
    else:
        for i, entry in enumerate(cache_entries, start=1):
            lines.append(f"### #{i}")
            lines.append("")
            lines.append(f"**Prompt:** `{entry.get('prompt', '')[:200]}`")
            lines.append("")
            lines.append("**Response:**")
            lines.append("")
            lines.append(entry.get("response", ""))
            lines.append("")
    lines.append("")

    # Feedback
    lines.append("## Recent feedback")
    lines.append("")
    recent_fb = (stats or {}).get("recent") or []
    if not recent_fb:
        lines.append("_(no feedback submitted)_")
    else:
        for fb in recent_fb[:25]:
            ts_raw = fb.get("ts", 0)
            try:
                ts = datetime.fromtimestamp(float(ts_raw), tz=timezone.utc).isoformat()
            except (TypeError, ValueError):
                ts = "unknown"
            rating = fb.get("rating", "?")
            comment = (fb.get("comment") or "").strip()
            cats = fb.get("categories") or []
            lines.append(
                f"- {ts} **rating:** `{rating}` "
                f"`trace={fb.get('trace_id', '')[:16]}…`"
                + (f" categories={cats}" if cats else "")
                + (f"\n  > {comment}" if comment else "")
            )
    lines.append("")

    # Raw JSON appendix — useful for programmatic consumers
    lines.append("## Appendix: raw JSON")
    lines.append("")
    lines.append("### export")
    lines.append("```json")
    lines.append(_json.dumps(export, indent=2))
    lines.append("```")
    if stats:
        lines.append("")
        lines.append("### feedback_stats")
        lines.append("```json")
        lines.append(_json.dumps(stats, indent=2))
        lines.append("```")
    lines.append("")

    return "\n".join(lines)


async def _handle_export() -> None:
    """Slash command: fetch + render the user's session as a download."""
    token = _user_token()
    if not token:
        await cl.Message(content="Not authenticated. Refresh and log in again.").send()
        return
    session_id = _thread_id()
    async with cl.Step(name="export", type="tool") as step:
        step.input = f"session_id={session_id[:8]}…"
        try:
            export = await _call_session_export(token, session_id)
            stats = await _call_feedback_stats(token, limit=50)
        except httpx.HTTPStatusError as e:
            step.is_error = True
            step.output = (
                f"upstream returned {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
            await cl.Message(content=f"❌ {step.output}").send()
            return
        except Exception as e:
            step.is_error = True
            step.output = f"{type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {step.output}").send()
            return

        md = _format_export_markdown(export, stats)
        step.output = (
            f"exported memory_turns={export.get('memory_turn_count', 0)} "
            f"cache_entries={export.get('cache_entry_count', 0)} "
            f"feedback_total={(stats or {}).get('total', 0)}"
        )

    # Render the markdown both inline (preview) and as a downloadable file.
    # cl.File takes content as bytes; encode UTF-8.
    safe_session = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:40]
    filename = f"session-export-{safe_session}.md"
    elements = [
        cl.File(
            name=filename,
            content=md.encode("utf-8"),
            display="inline",
            mime="text/markdown",
        ),
    ]
    preview_lines = md.splitlines()[:30]
    preview = "\n".join(preview_lines)
    if len(preview_lines) < len(md.splitlines()):
        preview += "\n\n_(truncated — download the file for the full export)_"
    await cl.Message(
        content=(
            f"📦 Session export ready. {export.get('memory_turn_count', 0)} memory "
            f"turns + {export.get('cache_entry_count', 0)} cache entries.\n\n"
            f"```markdown\n{preview}\n```"
        ),
        elements=elements,
    ).send()


async def _handle_delete_session() -> None:
    """Slash command: atomic wipe of per-(user, session) data."""
    token = _user_token()
    if not token:
        await cl.Message(content="Not authenticated. Refresh and log in again.").send()
        return
    session_id = _thread_id()
    async with cl.Step(name="delete-session", type="tool") as step:
        step.input = f"session_id={session_id[:8]}…"
        try:
            result = await _call_session_delete(token, session_id)
        except httpx.HTTPStatusError as e:
            step.is_error = True
            step.output = (
                f"upstream returned {e.response.status_code}: "
                f"{e.response.text[:200]}"
            )
            await cl.Message(content=f"❌ {step.output}").send()
            return
        except Exception as e:
            step.is_error = True
            step.output = f"{type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {step.output}").send()
            return

        deleted = result.get("deleted_keys", 0)
        step.output = f"deleted_keys={deleted}"

    await cl.Message(
        content=(
            f"🗑️ Session data wiped. Deleted **{result.get('deleted_keys', 0)}** "
            f"Redis keys for session `{session_id[:8]}…`. "
            f"Memory + cache for this session are gone.\n\n"
            f"_(Note: this does not clear the conversation history shown above "
            f"in this thread — that's local UI state. Reload the page or start "
            f"a new chat for a clean slate.)_"
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


# --- Phase #21: SSE streaming -----------------------------------------------
#
# /invoke/stream returns text/event-stream with four event types:
#   node_start  {"node": <name>}
#   node_end    {"node": <name>, ...selected fields}
#   token       {"content": <delta>}
#   done        full InvokeResponse JSON
#   error       {"status": int, "detail": str}
#
# Server format per event: a "event: <type>\n" line, one or more
# "data: <json>\n" lines, then a blank line.
#
# Chainlit pattern for streaming: cl.Message().stream_token() drips
# content into a single message bubble in real time. Steps open at
# node_start, close + render output at node_end. Done event finalizes
# the message + populates sources sidebar + Langfuse action.

# Per-node Step type — drives Chainlit's icon rendering.
_STEP_TYPE_BY_NODE = {
    "budget_check":         "tool",
    "input_validation":     "tool",
    "safety_input":         "tool",
    "pii_redact_input":     "tool",
    "cache_lookup":         "retrieval",
    "load_memory":          "retrieval",
    "rewrite_query":        "llm",
    "classify":             "llm",
    "retrieve":             "retrieval",
    "ensure_warm":          "tool",
    "plan":                 "llm",
    "execute":              "llm",
    "reflect":              "llm",
    "safety_output":        "tool",
    "hallucination_check":  "llm",
    "pii_redact_output":    "tool",
    "cache_store":          "retrieval",
    "save_memory":          "retrieval",
}


def _format_node_end(node: str, fields: Dict[str, Any]) -> str:
    """Render the per-node step output as compact markdown.

    Different nodes carry different interesting fields (cache_lookup
    has cache_hit + similarity; classify has route; safety_input has
    verdict + categories). Format each appropriately for the Chainlit
    Step body.
    """
    if node == "budget_check":
        return (
            f"action=`{fields.get('budget_action')}` "
            f"consumed={fields.get('budget_consumed')} "
            f"remaining={fields.get('budget_remaining')}"
        )
    if node == "input_validation":
        details = fields.get("input_validation_details", {}) or {}
        return f"action=`{fields.get('input_validation_action')}` details={details}"
    if node == "safety_input":
        verdict = fields.get("safety_input_verdict")
        cats = fields.get("safety_categories") or []
        return f"verdict=`{verdict}` categories={cats}"
    if node == "cache_lookup":
        hit = fields.get("cache_hit")
        sim = fields.get("cache_similarity", 0.0)
        return f"hit=`{hit}` similarity={sim:.4f}"
    if node == "classify":
        return f"route=`{fields.get('route')}` raw=`{fields.get('classifier_raw', '')[:80]}`"
    if node == "retrieve":
        return (
            f"chunks={fields.get('retrieve_count', 0)} "
            f"in {fields.get('retrieve_ms', 0)}ms"
        )
    if node == "ensure_warm":
        cold = fields.get("cold_start")
        if cold:
            return f"❄️ cold start, waited {fields.get('warm_wait_seconds', 0):.1f}s"
        return "✅ already warm"
    if node == "plan":
        return f"action=`{fields.get('planner_action')}` steps={fields.get('plan_steps_count', 0)}"
    if node == "execute":
        return (
            f"latency={fields.get('execute_latency_ms', 0)}ms "
            f"tools={fields.get('tool_calls_log') or []}"
        )
    if node == "reflect":
        return (
            f"cycles={fields.get('cycles', 0)} "
            f"needs_more={fields.get('needs_more_context', False)}"
        )
    if node == "safety_output":
        return f"verdict=`{fields.get('safety_output_verdict')}` action=`{fields.get('safety_action')}`"
    if node == "hallucination_check":
        return (
            f"action=`{fields.get('hallucination_action')}` "
            f"verdict=`{fields.get('hallucination_verdict')}` "
            f"confidence={fields.get('hallucination_confidence', 0):.2f}"
        )
    if node == "pii_redact_output":
        ents = fields.get("pii_entities_found") or {}
        return f"action=`{fields.get('pii_redact_action')}` entities={ents}"
    return ""


async def _stream_invoke(
    prompt: str, token: str, session_id: str
):
    """Async generator yielding (event_type, payload_dict) from /invoke/stream.

    Parses the SSE wire format (event: / data: lines, blank-line
    delimiters). Each yielded tuple corresponds to one SSE frame.
    """
    async with httpx.AsyncClient(verify=False, timeout=600.0) as client:
        async with client.stream(
            "POST",
            f"{LANGGRAPH_URL}/invoke/stream",
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "accept": "text/event-stream",
            },
            json={
                "prompt":     prompt,
                "max_tokens": 600,
                "session_id": session_id,
                "top_k":      5,
            },
        ) as resp:
            resp.raise_for_status()
            event_type: str = ""
            data_lines: list = []
            async for line in resp.aiter_lines():
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line == "":
                    # Blank line = end of frame. Emit if both halves
                    # are present.
                    if event_type and data_lines:
                        try:
                            payload = json.loads("\n".join(data_lines))
                            yield (event_type, payload)
                        except json.JSONDecodeError:
                            pass
                    event_type = ""
                    data_lines = []


async def _on_message_stream(
    prompt: str, token: str, session_id: str, t0: float
) -> None:
    """Stream-consume /invoke/stream and render progressively."""
    final_data: Dict[str, Any] = {}
    open_steps: Dict[str, Any] = {}
    error_payload: Dict[str, Any] = {}

    async with cl.Step(name="LangGraph router", type="run") as outer:
        outer.input = prompt

        # Empty assistant bubble we'll stream tokens into.
        msg = cl.Message(content="")
        await msg.send()

        try:
            async for ev_type, ev_data in _stream_invoke(prompt, token, session_id):
                if ev_type == "node_start":
                    node = ev_data.get("node", "")
                    if not node:
                        continue
                    step = cl.Step(
                        name=node,
                        type=_STEP_TYPE_BY_NODE.get(node, "tool"),
                        parent_id=outer.id,
                    )
                    await step.send()
                    open_steps[node] = step

                elif ev_type == "node_end":
                    node = ev_data.get("node", "")
                    if not node:
                        continue
                    step = open_steps.pop(node, None)
                    if step is not None:
                        step.output = _format_node_end(node, ev_data) or " "
                        await step.update()

                elif ev_type == "token":
                    content = ev_data.get("content", "")
                    if content:
                        await msg.stream_token(content)

                elif ev_type == "done":
                    final_data = ev_data
                    break

                elif ev_type == "error":
                    error_payload = ev_data
                    break

        except httpx.HTTPStatusError as e:
            outer.is_error = True
            # Streaming responses (httpx.AsyncClient.stream) require
            # the body to be explicitly read before .text/.content
            # is accessible. Touching e.response.text directly here
            # raised httpx.ResponseNotRead and masked the real status
            # code in the user-visible error message — caught during
            # the Phase #45 canary smoke when langgraph-service
            # returned 401. Calling aread() first unblocks .text;
            # wrap in try/except so a transport-level failure (no
            # body to read) still surfaces the status code.
            try:
                body = (await e.response.aread()).decode(errors="replace")[:200]
            except Exception:
                body = "<failed to read response body>"
            outer.output = f"upstream returned {e.response.status_code}: {body}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return
        except Exception as e:
            outer.is_error = True
            outer.output = f"{type(e).__name__}: {e}"
            await cl.Message(content=f"❌ {outer.output}").send()
            return

        if error_payload:
            outer.is_error = True
            outer.output = (
                f"upstream error: {error_payload.get('detail', 'unknown')}"
            )
            await cl.Message(content=f"❌ {outer.output}").send()
            return

        # Some safe-block paths short-circuit before tokens flow —
        # populate the message from final_data.response if the bubble
        # is still empty.
        if not msg.content and final_data.get("response"):
            msg.content = final_data["response"]
            await msg.update()

        # Outer step summary line — rolls up the final state.
        outer.output = (
            f"route={final_data.get('route')}  "
            f"retrieved={final_data.get('retrieve_count', 0)}  "
            f"cold_start={final_data.get('cold_start', False)}  "
            f"latency={final_data.get('execute_latency_ms', 0)}ms  "
            f"total={int((time.time() - t0) * 1000)}ms  "
            f"cache_hit={final_data.get('cache_hit', False)}"
        )

    # Sources sidebar + Langfuse action — same shape as the sync
    # handler builds.
    elements: list = [
        cl.Text(
            name="routing.json",
            content=json.dumps(final_data, indent=2),
            display="side",
        ),
    ]
    if final_data.get("retrieve_count", 0) > 0:
        sources_md_lines = ["# Sources used as RAG context", ""]
        for i, c in enumerate(final_data.get("retrieved_chunks", []), start=1):
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

    actions: list = []
    trace_id = final_data.get("langfuse_trace_id")
    if trace_id:
        actions.append(
            cl.Action(
                name="view_in_langfuse",
                payload={"trace_id": trace_id},
                label="🔍 View in Langfuse",
                tooltip="Open the matching trace in Langfuse",
            )
        )

    # Attach elements + actions to the streamed message we already
    # sent. Chainlit's msg.update() picks up the new fields.
    msg.elements = elements
    msg.actions = actions
    await msg.update()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    cmd = message.content.strip().lower()
    if cmd in ("/upload", "upload"):
        await _handle_upload()
        return
    if cmd in ("/export", "export"):
        await _handle_export()
        return
    if cmd in ("/delete-session", "/forget"):
        await _handle_delete_session()
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

    # Phase #21: route to streaming or sync handler. Streaming gives
    # token-by-token UX + node progress as it happens; sync remains
    # for fallback debugging (or if SSE ever proves unreliable through
    # an intermediate proxy).
    if STREAM_ENABLED:
        await _on_message_stream(prompt, token, session_id, t0)
        return

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
