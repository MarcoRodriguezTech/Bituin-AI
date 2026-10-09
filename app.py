"""Bituin AI — privacy-first local assistant for your Gmail inbox.

Run:  streamlit run app.py
"""
import datetime
import re
import socket
import subprocess
from pathlib import Path

import streamlit as st

from config.settings import LLM_MODEL, OLLAMA_HOST
from src.connectors.gmail_client import GmailClient
from src.database.db_manager import DBManager

try:
    from src.rag.llm_chain import RAGPipeline

    _RAG_IMPORT_ERR = None
except Exception as _e:  # noqa: BLE001 — dashboard still renders without RAG deps
    RAGPipeline = None
    _RAG_IMPORT_ERR = _e

ASSETS = Path(__file__).parent / "assets"
LOGO = ASSETS / "Bituin AI flipped.png"
LOGO = LOGO if LOGO.exists() else ASSETS / "Bituin AI.png"
USER_AVATAR = ASSETS / "avatar-user.png"

st.set_page_config(
    page_title="Bituin AI",
    page_icon=str(LOGO) if LOGO.exists() else "✦",
    layout="wide",
)


# ---------- Cached resources ----------
@st.cache_resource
def get_db() -> DBManager:
    return DBManager()


@st.cache_resource
def get_pipeline():
    if _RAG_IMPORT_ERR is not None:
        return _RAG_IMPORT_ERR
    try:
        return RAGPipeline()
    except Exception as e:  # noqa: BLE001
        return e


@st.cache_resource
def get_voice():
    try:
        from src.voice.whisper_listener import VoiceDictator

        return VoiceDictator()
    except Exception as e:  # noqa: BLE001
        return e


@st.cache_resource
def get_analyzer():
    try:
        from src.rag.analyzer import EmailAnalyzer

        return EmailAnalyzer()
    except Exception as e:  # noqa: BLE001
        return e


_URGENCY_BADGE = {
    "urgent": ":red-badge[:material/priority_high: Urgent]",
    "action": ":orange-badge[:material/notifications_active: Action]",
    "fyi": ":gray-badge[:material/draft: FYI]",
}
_URGENCY_ORDER = {"urgent": 0, "action": 1, "fyi": 2}

_SUGGESTIONS = {
    ":red[:material/priority_high:] What's urgent?": "What emails need my attention right now?",
    ":orange[:material/event:] Deadlines?": "Summarize my upcoming deadlines and due dates",
    ":violet[:material/mail:] Digest": "Give me a quick digest of my unread inbox",
}

# Intent routing — keeps answers consistent:
# - summary/digest questions get deterministic "recent messages" context by date,
#   not whatever vector search happens to surface
# - reformat/follow-up questions reuse the previous answer's sources
_SUMMARY_RE = re.compile(
    r"(summar|digest|recap|catch me up|what'?s new|overview|inbox|unread|"
    r"today|yesterday|this week|recent|latest|new emails?)",
    re.I,
)
_FOLLOWUP_RE = re.compile(
    r"^(put|make|reformat|rephrase|rewrite|redo|retry|re-?run|same|"
    r"that|this|those|these|them|the last|the previous|try again|"
    r"once more|do (it|that)|shorter|longer|simpler|in one|in a|"
    r"in paragraph|bullet|as a|repeat|again|why|expand|elaborate|"
    r"can you|could you|thanks|ok(ay)?\b|yes\b|no\b|and |but |also )",
    re.I,
)


def _records_as_hits(recs: list[dict]) -> list[dict]:
    """Shape DB records like vector-search hits for the RAG chain."""
    return [{"metadata": r, "content": r["content"]} for r in recs]


def _cluster_hits(recs: list[dict]) -> list[dict]:
    """Collapse near-duplicate threads (e.g. repeated CI notifications) into a
    single representative hit so summaries cover the whole inbox variety."""
    groups: dict[tuple, list[dict]] = {}
    order: list[tuple] = []
    for r in recs:
        subj = re.sub(r"\s*\([0-9a-f]{5,}\)\s*$", "", r["subject"])
        subj = re.sub(r"is live:.*$", "is live", subj, flags=re.I)
        subj = re.sub(r"\s+-.*$", "", subj).strip().lower()
        key = (r["sender"].split("@")[-1].lower(), subj)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)
    hits = []
    for key in order:
        g = groups[key]
        rep = dict(g[0])
        if len(g) > 1:
            rep["subject"] = f"[{len(g)} emails] {g[0]['subject']}"
            rep["content"] = g[0]["content"] + "\nSimilar subjects: " + "; ".join(
                r["subject"] for r in g[1:6]
            )
        hits.append({"metadata": rep, "content": rep["content"]})
    return hits


# ---------- Status helpers ----------
def ollama_alive() -> bool:
    try:
        import ollama

        ollama.Client(host=OLLAMA_HOST).list()
        return True
    except Exception:  # noqa: BLE001
        return False


def vram_usage() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            timeout=5,
            text=True,
        ).strip()
        used, total = out.split(",")
        return f"{int(used.strip()) / 1024:.1f} / {int(total.strip()) / 1024:.1f} GB"
    except Exception:  # noqa: BLE001
        return "n/a"


def online() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2).close()
        return True
    except OSError:
        return False


# ---------- UI ----------
_head_img, _head_txt = st.columns([1, 4], gap="small", vertical_alignment="center")
with _head_img:
    if LOGO.exists():
        st.image(str(LOGO), width="stretch")
with _head_txt:
    st.title("Bituin AI")
    st.caption("Your inbox, understood. Local AI — nothing leaves this machine.")

db = get_db()
pipeline = get_pipeline()
pipe_ok = not isinstance(pipeline, Exception)
vs = pipeline.vector_store if pipe_ok else None
analyzer = get_analyzer()
an_ok = not isinstance(analyzer, Exception)
voice = get_voice()
voice_ok = not isinstance(voice, Exception)
oll_ok = ollama_alive()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Bituin AI", icon=":material/nights_stay:")
    st.caption("Local-first inbox assistant")
    st.divider()
    st.header("Sync", icon=":material/sync:")
    if st.button("Sync Gmail inbox", icon=":material/refresh:", width="stretch"):
        with st.status("Syncing…", expanded=True) as status:
            st.write("Fetching unread email")
            records = GmailClient().fetch()
            new = db.insert_many(records)
            chunks = vs.index_records(db.fetch_all()) if vs else 0
            st.write(f"{new} new · {chunks} chunks indexed")

            pending = db.unanalyzed_ids()
            if pending and an_ok and oll_ok:
                rec_map = {r["id"]: r for r in db.fetch_all()}
                prog = st.progress(0.0)
                for i, eid in enumerate(pending):
                    res = analyzer.analyze(rec_map[eid])
                    db.save_analysis(eid, res["urgency"], res["reason"], res["tasks"])
                    prog.progress((i + 1) / len(pending))
                st.write(f"Analyzed {len(pending)} emails")
            status.update(label="Sync complete", state="complete", expanded=False)

    st.divider()
    counts = db.urgency_counts()
    c1, c2 = st.columns(2)
    c1.metric("Messages", db.count())
    c2.metric("Urgent", counts.get("urgent", 0))
    st.metric("Open tasks", len(db.fetch_tasks(include_done=False)))

    st.divider()
    st.subheader("System", icon=":material/memory:")
    st.markdown(
        f"{'🟢' if oll_ok else '🔴'} LLM `{LLM_MODEL}`  \n"
        f"{'🟢' if voice_ok else '🔴'} Whisper `base.en`  \n"
        f":material/memory_alt: VRAM `{vram_usage()}`  \n"
        f"{'🟢 Online' if online() else '🌙 Offline — fully local'}"
    )

# ---------- Main tabs ----------
amap = db.fetch_analysis_map()
tab_ask, tab_todo, tab_inbox, tab_digest = st.tabs(
    [
        ":material/chat: Ask",
        ":material/checklist: To-do",
        f":material/inbox: Inbox ({db.count()})",
        ":material/wb_twilight: Digest",
    ]
)

# ---------- Ask ----------
with tab_ask:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.caption("Ask anything about your unread inbox — or tap the mic.")
        sel = st.pills(
            "Suggestions", list(_SUGGESTIONS.keys()), label_visibility="collapsed"
        )
    else:
        sel = None

    for msg in st.session_state.messages:
        avatar = str(USER_AVATAR) if msg["role"] == "user" else str(LOGO)
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            srcs = msg.get("sources", [])
            if srcs:
                with st.popover(
                    f"References · {len(srcs)}", icon=":material/attachment:"
                ):
                    for hit in srcs:
                        m = hit["metadata"]
                        st.markdown(f"- **{m['subject']}** — {m['sender']}")

    prompt = st.chat_input(
        "Ask about your inbox…",
        accept_audio=voice_ok,
        submit_mode="stop",
    )

    pending_prompt = None
    if sel:
        pending_prompt = _SUGGESTIONS[sel]
    elif prompt:
        if prompt.audio and not prompt.text:
            pending_prompt = voice.transcribe_bytes(prompt.audio.getvalue())
        else:
            pending_prompt = prompt.text

    if pending_prompt:
        st.session_state.messages.append({"role": "user", "content": pending_prompt})
        with st.chat_message("user", avatar=str(USER_AVATAR)):
            st.markdown(pending_prompt)

        with st.chat_message("assistant", avatar=str(LOGO)):
            if not pipe_ok:
                st.error(f"Pipeline unavailable: {pipeline}")
                answer, sources = "", []
            elif not oll_ok:
                st.error(f"Ollama offline — start it at {OLLAMA_HOST}")
                answer, sources = "", []
            else:
                with st.status(":shimmer[Searching inbox…]", type="compact") as status:
                    prev_sources = next(
                        (
                            m.get("sources")
                            for m in reversed(st.session_state.messages[:-1])
                            if m["role"] == "assistant" and m.get("sources")
                        ),
                        [],
                    )
                    if prev_sources and _FOLLOWUP_RE.match(pending_prompt.strip()):
                        sources = prev_sources
                        how = "same context as before"
                    elif _SUMMARY_RE.search(pending_prompt):
                        sources = _cluster_hits(db.fetch_recent(days=2, limit=40))
                        if not sources:
                            sources = pipeline.retrieve(pending_prompt)
                        how = "recent messages, grouped"
                    else:
                        query = pending_prompt
                        prev_user = next(
                            (
                                m["content"]
                                for m in reversed(st.session_state.messages[:-1])
                                if m["role"] == "user"
                            ),
                            "",
                        )
                        if prev_user and len(pending_prompt.split()) < 6:
                            query = f"{prev_user} {pending_prompt}"
                        sources = pipeline.retrieve(query)
                        how = "relevant messages"
                    for hit in sources:
                        m = hit["metadata"]
                        st.markdown(f"- :material/mail: **{m['subject']}** — {m['sender']}")
                    status.update(
                        label=f"{len(sources)} {how}", state="complete"
                    )
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                answer = st.write_stream(
                    pipeline.stream_query(pending_prompt, sources, history=history)
                )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()

# ---------- To-do ----------
with tab_todo:
    tasks = db.fetch_tasks()
    st.caption(
        "The ✕ button only removes a task from this list — it does not delete "
        "or modify the email in your inbox."
    )
    if tasks:
        st.caption(f"{len(db.fetch_tasks(include_done=False))} open · extracted from your email")
        with st.expander("Remove multiple tasks", icon=":material/delete_sweep:"):
            remove_ids = st.multiselect(
                "Select tasks to remove",
                options=[t["id"] for t in tasks],
                format_func=lambda i: next(
                    (t["description"] for t in tasks if t["id"] == i), i
                )[:80],
                label_visibility="collapsed",
                placeholder="Pick tasks…",
            )
            if st.button(
                f"Remove {len(remove_ids)} selected" if remove_ids else "Remove selected",
                icon=":material/delete:",
                disabled=not remove_ids,
            ):
                for tid in remove_ids:
                    db.delete_task(tid)
                st.rerun()
    else:
        st.caption("No tasks yet — sync your inbox and Bituin extracts the action items.")
    for t in tasks:
        due = ""
        if t["due_date"]:
            try:
                days = (
                    datetime.date.fromisoformat(str(t["due_date"])[:10])
                    - datetime.date.today()
                ).days
                due = f"  :gray[· due {t['due_date'][:10]}]"
                if days < 0:
                    due += " :red[overdue]"
                elif days == 0:
                    due += " :orange[today]"
                else:
                    due += f" :gray[in {days}d]"
            except ValueError:
                due = f"  :gray[· due {t['due_date']}]"
        with st.container(horizontal=True, vertical_alignment="center"):
            checked = st.checkbox(
                f"{t['description']}{due}",
                value=t["done"],
                key=t["id"],
            )
            if st.button(
                "Remove",
                key=f"del-{t['id']}",
                icon=":material/close:",
                type="tertiary",
            ):
                db.delete_task(t["id"])
                st.rerun()
            if checked != t["done"]:
                db.set_task_done(t["id"], checked)
                st.rerun()
        st.caption(f":material/mail: {t['subject']} — {t['sender']}")

# ---------- Inbox ----------
with tab_inbox:
    emails = sorted(
        db.fetch_all(),
        key=lambda r: _URGENCY_ORDER.get(amap.get(r["id"], {}).get("urgency"), 3),
    )
    if not emails:
        st.caption("Inbox empty — hit Sync in the sidebar.")
    for rec in emails:
        a = amap.get(rec["id"])
        badge = _URGENCY_BADGE.get(a["urgency"], "") if a else ""
        with st.container(border=True):
            st.markdown(f"{badge}  **{rec['subject']}**")
            st.caption(f"{rec['sender']} · {rec['timestamp'][:10]}")
            if a and a["reason"]:
                st.markdown(f":gray[{a['reason']}]")

# ---------- Digest ----------
with tab_digest:
    st.caption("One click. A briefing written entirely on this machine.")
    if st.button(
        "Generate tonight's briefing", icon=":material/wb_twilight:", type="primary"
    ):
        if not an_ok or not oll_ok:
            st.error("Ollama is required for the briefing.")
        else:
            with st.spinner("Reading the stars…"):
                st.session_state["digest"] = analyzer.digest(
                    db.fetch_all(), amap, db.fetch_tasks(include_done=False)
                )
    if st.session_state.get("digest"):
        with st.container(border=True):
            st.markdown(st.session_state["digest"])
