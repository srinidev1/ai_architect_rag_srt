import streamlit as st
from pathlib import Path
from implementation.answer import answer_question
from langchain_core.documents import Document

DB_NAME = str(Path(__file__).parent.parent / "vector_db")


def _init_session():
    """Bootstrap session state keys used by this view."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []       # list[{"role": str, "content": str}]
    if "source_docs" not in st.session_state:
        st.session_state["source_docs"] = []    # list[Document] from last answer


def _render_sources(docs: list[Document]) -> None:
    """Show retrieved source documents in the sidebar."""
    if not docs:
        return
    st.sidebar.divider()
    st.sidebar.markdown("### 📄 Source Documents")
    for i, doc in enumerate(docs, 1):
        with st.sidebar.expander(f"Source {i}"):
            st.markdown(doc.page_content)
            if doc.metadata:
                st.caption(str(doc.metadata))


def _render_history() -> None:
    """Replay all previous turns as chat bubbles."""
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _handle_question(question: str) -> None:
    """
    Append the user message, call answer_question, append the assistant reply,
    and stash retrieved docs — all in session state.
    """
    # 1. Show + persist user bubble
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. Call RAG backend
    # Pass history *excluding* the message we just appended so the backend
    # doesn't double-count the current question when building combined_question.
    history = st.session_state["messages"][:-1]

    userrole = st.session_state.get("role", "unknown")

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer, docs = answer_question(question=question, userrole=userrole, history=history)
        st.markdown(answer)

    # 3. Persist answer + docs
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.session_state["source_docs"] = docs

    # 4. Rerun so the sidebar refreshes with the new sources
    st.rerun()


def render():
    _init_session()

    st.title("💬 Chat")

    # Guard: knowledge base must exist before we can chat
    if not Path(DB_NAME).exists():
        st.warning("No knowledge base found. Please run the Ingest process first.")
        return

    st.caption("Chat with your AI assistant powered by Insurellm knowledge base.")

    # Sidebar: clear history + source docs
    with st.sidebar:
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["source_docs"] = []
            st.rerun()

    # Replay existing conversation
    _render_history()

    # Source docs from the last answer
    _render_sources(st.session_state["source_docs"])

    # ── Input ──────────────────────────────────────────────────────────────────
    # Primary: bottom chat bar (feels like ChatGPT)
    user_input = st.chat_input("Ask me anything about Insurellm…")
    if user_input and user_input.strip():
        _handle_question(user_input.strip())
