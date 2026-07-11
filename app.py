"""
app.py
------
Streamlit chat interface for the GigaCorp FAQ RAG assistant.

Run with:
    streamlit run app.py
"""

import os
import warnings
import streamlit as st

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="GigaCorp Support Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark-glass look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Page background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1040 50%, #141e30 100%);
        min-height: 100vh;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 6rem;
        max-width: 820px;
    }

    /* ── Title bar ── */
    .gc-title {
        text-align: center;
        padding: 1.6rem 0 0.4rem;
    }
    .gc-title h1 {
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .gc-title p {
        color: #94a3b8;
        font-size: 0.9rem;
        margin: 0.35rem 0 0;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        backdrop-filter: blur(10px);
    }

    /* ── User bubble accent ── */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        border-color: rgba(167,139,250,0.3);
        background: rgba(167,139,250,0.07);
    }

    /* ── Input box ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(167,139,250,0.6) !important;
        box-shadow: 0 0 0 2px rgba(167,139,250,0.15) !important;
    }

    /* ── Expander (Sources) ── */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 10px !important;
    }
    [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.03);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 2rem; }

    /* ── Reset button ── */
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        font-size: 0.88rem;
        cursor: pointer;
        transition: opacity 0.2s;
        letter-spacing: 0.2px;
    }
    div[data-testid="stSidebar"] .stButton > button:hover { opacity: 0.85; }

    /* ── Source badge ── */
    .src-badge {
        display: inline-block;
        background: rgba(96,165,250,0.15);
        border: 1px solid rgba(96,165,250,0.3);
        border-radius: 6px;
        padding: 0.15rem 0.5rem;
        font-size: 0.78rem;
        color: #93c5fd;
        font-family: 'Inter', monospace;
        margin: 0.2rem 0.2rem 0 0;
    }

    /* ── Rewritten query pill ── */
    .rewritten-pill {
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.25);
        border-radius: 8px;
        padding: 0.3rem 0.7rem;
        font-size: 0.78rem;
        color: #6ee7b7;
        margin-bottom: 0.5rem;
        display: inline-block;
    }

    /* ── Typing indicator ── */
    .typing-indicator { color: #94a3b8; font-size: 0.85rem; }

    /* ── Divider ── */
    hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# @st.cache_resource — builds shared resources once per server process
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading GigaCorp knowledge base...")
def get_session_factory():
    from rag_chain import build_session_factory
    return build_session_factory()

# ---------------------------------------------------------------------------
# Per-session state initialisation
# ---------------------------------------------------------------------------
def init_state() -> None:
    factory = get_session_factory()

    if "gc_session" not in st.session_state:
        st.session_state.gc_session = factory()

    if "messages" not in st.session_state:
        # Each entry: {"role": "user"|"assistant", "content": str,
        #              "rewritten": str|None, "chunks": list|None}
        st.session_state.messages = []

    if "session_factory" not in st.session_state:
        st.session_state.session_factory = factory


def reset_conversation() -> None:
    factory = st.session_state.session_factory
    st.session_state.gc_session = factory()       # fresh GigaCorpSession
    st.session_state.messages   = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding-bottom:1rem;'>
                <span style='font-size:2.2rem;'>🤖</span>
                <h3 style='color:#e2e8f0; margin:0.4rem 0 0; font-size:1.1rem;'>
                    GigaCorp Assistant
                </h3>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🔄  Reset conversation", key="reset_btn"):
            reset_conversation()
            st.rerun()

        st.markdown("---")

        st.markdown(
            """
            <div style='color:#94a3b8; font-size:0.82rem; line-height:1.6;'>
            <b style='color:#c4b5fd;'>ℹ️ About this demo</b><br><br>
            This assistant answers questions about <b>GigaCorp</b>, a
            <em>fictional company</em> created for demonstration purposes.<br><br>
            All policies, prices, contacts, and hours are <b>mock data</b>
            sourced from <code>data/gigacorp_faq.txt</code>.<br><br>
            <b style='color:#c4b5fd;'>How it works</b><br><br>
            1. Your question is <em>condensed</em> using chat history<br>
            2. The top-3 relevant FAQ chunks are retrieved from a
               local FAISS index<br>
            3. <b>Llama 3.3 70B</b> (via Groq) answers using only
               those chunks<br>
            4. Every claim is cited to the exact source lines<br><br>
            <hr style='border-color:rgba(255,255,255,0.08);'>
            <span style='color:#64748b;'>Model: llama-3.3-70b-versatile<br>
            Embeddings: all-MiniLM-L6-v2</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            "<p style='color:#64748b; font-size:0.75rem; text-align:center;'>"
            "GigaCorp &copy; 2026 &nbsp;|&nbsp; Demo only"
            "</p>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Render a single chat message (history replay)
# ---------------------------------------------------------------------------
def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant":
            # Show rewritten query pill if applicable
            if msg.get("rewritten") and msg["rewritten"] != msg.get("original_query", ""):
                st.markdown(
                    f'<span class="rewritten-pill">'
                    f'🔍 Interpreted as: <em>{msg["rewritten"]}</em>'
                    f'</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])

            # Sources expander
            chunks = msg.get("chunks") or []
            cited  = msg.get("cited_lines") or []
            if chunks or cited:
                with st.expander("📄 Sources", expanded=False):
                    if cited:
                        badges = "".join(
                            f'<span class="src-badge">gigacorp_faq.txt '
                            f'lines {s}–{e}</span>'
                            for s, e in cited
                        )
                        st.markdown(badges, unsafe_allow_html=True)

                    if chunks:
                        st.markdown("")
                        for i, chunk in enumerate(chunks, 1):
                            with st.container():
                                st.markdown(
                                    f"**#{i} · {chunk.get('qa_id', 'N/A')} · "
                                    f"{chunk.get('section', '')}**  \n"
                                    f"`{chunk.get('citation', '')}`"
                                )
        else:
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    try:
        init_state()
    except Exception as e:
        render_sidebar()
        st.markdown(
            """
            <div class="gc-title">
                <h1>GigaCorp Support Assistant</h1>
                <p>System Configuration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.error(
            f"### ⚠️ Configuration Error\n\n"
            f"**Error Details:** {str(e)}\n\n"
            "To resolve this, please configure your Groq API Key:\n\n"
            "1. **Locally:** Create a `.env` file in the root folder with `GROQ_API_KEY=gsk_...`\n"
            "2. **Streamlit Cloud:** Add `GROQ_API_KEY = \"gsk_...\"` in the **Secrets** manager of your Streamlit dashboard (lower-right corner -> Settings -> Secrets)."
        )
        st.stop()

    render_sidebar()

    # ── Title ──────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="gc-title">
            <h1>GigaCorp Support Assistant</h1>
            <p>Ask anything about shipping, returns, business hours, or service plans.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Welcome message (first visit) ──────────────────────────────────────
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Hello! I'm the **GigaCorp Support Assistant**. "
                "I can answer questions about our shipping policies, return process, "
                "business hours, and service tiers.\n\n"
                "Try asking something like:\n"
                "- *Do you ship internationally?*\n"
                "- *What is your return window?*\n"
                "- *What's included in the Professional plan?*"
            )

    # ── Replay history ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        render_message(msg)

    # ── Chat input ─────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about GigaCorp…"):
        # 1. Show user bubble
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # 2. Run RAG chain + show assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    response = st.session_state.gc_session.ask(prompt)
                except Exception as exc:
                    st.error(f"An error occurred: {exc}")
                    st.stop()

            # Rewritten query pill
            if response.rewritten_query != prompt:
                st.markdown(
                    f'<span class="rewritten-pill">'
                    f'🔍 Interpreted as: <em>{response.rewritten_query}</em>'
                    f'</span>',
                    unsafe_allow_html=True,
                )

            st.markdown(response.answer)

            # Sources expander
            if response.source_chunks or response.cited_lines:
                with st.expander("📄 Sources", expanded=False):
                    if response.cited_lines:
                        badges = "".join(
                            f'<span class="src-badge">gigacorp_faq.txt '
                            f'lines {s}–{e}</span>'
                            for s, e in response.cited_lines
                        )
                        st.markdown(badges, unsafe_allow_html=True)

                    if response.source_chunks:
                        st.markdown("")
                        for i, chunk in enumerate(response.source_chunks, 1):
                            st.markdown(
                                f"**#{i} · {chunk.get('qa_id', 'N/A')} · "
                                f"{chunk.get('section', '')}**  \n"
                                f"`{chunk.get('citation', '')}`"
                            )

        # 3. Persist to session_state for history replay
        st.session_state.messages.append(
            {
                "role":           "assistant",
                "content":        response.answer,
                "original_query": prompt,
                "rewritten":      response.rewritten_query,
                "cited_lines":    response.cited_lines,
                "chunks":         response.source_chunks,
            }
        )


if __name__ == "__main__":
    main()
