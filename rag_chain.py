"""
rag_chain.py
------------
Conversational RAG chain for the GigaCorp FAQ.

Architecture (per turn):
  1. CONDENSE  – LLM rewrites the follow-up question into a self-contained
                 standalone query using the conversation history.
  2. RETRIEVE  – Top-3 FAISS chunks fetched using the standalone query.
  3. ANSWER    – LLM generates a grounded, cited answer from those chunks.
  4. MEMORY    – HumanMessage / AIMessage pairs are appended to an in-memory
                 ChatMessageHistory that persists for the session's lifetime.

Usage (interactive REPL):
    python rag_chain.py

Usage (single-shot CLI):
    python rag_chain.py "Do you ship to India?"
"""

import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = Path(__file__).parent.resolve() / "vectorstore"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
TOP_K           = 3

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# Step 1 – Condense follow-up into a standalone question
CONDENSE_SYSTEM = (
    "You are a query rewriter. "
    "Given a conversation history and a follow-up question, rewrite the "
    "follow-up into a single, self-contained question that captures all "
    "necessary context from the history. "
    "If the question is already self-contained, return it unchanged. "
    "Output ONLY the rewritten question — no explanation, no preamble."
)

CONDENSE_HUMAN_TMPL = """\
Conversation history:
{history}

Follow-up question: {question}

Standalone question:"""

# Step 2 – Grounded, cited answer
ANSWER_SYSTEM_TMPL = """\
You are GigaCorp's customer support assistant.
Answer ONLY using the context passages below. Do NOT use outside knowledge.

Rules:
1. Base every factual claim strictly on the provided context.
2. After each claim, cite the source using the exact format: [lines X-Y].
   The line numbers appear in each context block's header — use them exactly.
3. If the answer is not in the context, respond with:
   "I'm sorry, I don't have that information in GigaCorp's documentation."
4. Never invent policies, prices, dates, or contact details.
5. Keep your answer concise and professional.

Context:
{context}"""

# ---------------------------------------------------------------------------
# Structured response
# ---------------------------------------------------------------------------
@dataclass
class RAGResponse:
    query:            str            # original user input
    rewritten_query:  str            # standalone query sent to retriever
    answer:           str            # LLM-generated answer
    cited_lines:      list[tuple[int, int]] = field(default_factory=list)
    source_chunks:    list[dict]            = field(default_factory=list)

    def pretty(self) -> str:
        parts = [f"Query    : {self.query}"]
        if self.rewritten_query != self.query:
            parts.append(f"Rewritten: {self.rewritten_query}")
        parts.append(f"Answer   : {self.answer}")
        if self.cited_lines:
            ranges = ", ".join(f"lines {s}-{e}" for s, e in self.cited_lines)
            parts.append(f"Sources  : gigacorp_faq.txt [{ranges}]")
        else:
            parts.append("Sources  : (no specific lines cited)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_CITATION_RE = re.compile(r"\[lines\s+(\d+)\s*[-\u2013]\s*(\d+)\]", re.IGNORECASE)

def extract_cited_lines(text: str) -> list[tuple[int, int]]:
    seen, pairs = set(), []
    for m in _CITATION_RE.finditer(text):
        pair = (int(m.group(1)), int(m.group(2)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def history_to_text(messages) -> str:
    """Convert a list of BaseMessage objects to a readable transcript."""
    lines = []
    for msg in messages:
        role = "User" if msg.type == "human" else "Assistant"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines) if lines else "(none)"


# ---------------------------------------------------------------------------
# Session class — wraps the chain + its own message history
# ---------------------------------------------------------------------------
class GigaCorpSession:
    """
    One conversational session.  Keeps its own in-memory history and exposes
    an `ask(query)` method that runs all three chain steps.
    """

    def __init__(self, retriever, llm):
        from langchain_community.chat_message_histories import ChatMessageHistory
        self._retriever = retriever
        self._llm       = llm
        self._history   = ChatMessageHistory()

    # ------------------------------------------------------------------
    # Step 1 — Condense follow-up into standalone question
    # ------------------------------------------------------------------
    def _condense(self, question: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage

        history_text = history_to_text(self._history.messages)

        # No history yet → question is already standalone
        if not self._history.messages:
            return question

        messages = [
            SystemMessage(content=CONDENSE_SYSTEM),
            HumanMessage(
                content=CONDENSE_HUMAN_TMPL.format(
                    history=history_text,
                    question=question,
                )
            ),
        ]
        result = self._llm.invoke(messages)
        standalone = result.content.strip().strip('"').strip("'")
        return standalone if standalone else question

    # ------------------------------------------------------------------
    # Step 2 — Retrieve top-K chunks
    # ------------------------------------------------------------------
    def _retrieve(self, standalone_query: str):
        return self._retriever.invoke(standalone_query)

    # ------------------------------------------------------------------
    # Step 3 — Generate grounded answer
    # ------------------------------------------------------------------
    def _answer(self, standalone_query: str, docs) -> tuple[str, list[dict]]:
        from langchain_core.messages import SystemMessage, HumanMessage

        context_blocks, chunk_meta = [], []
        for doc in docs:
            m = doc.metadata
            header = (
                f"[Source: {m.get('citation', 'unknown')} | "
                f"Section: {m.get('section', '')} | "
                f"QA: {m.get('qa_id', '')}]"
            )
            context_blocks.append(f"{header}\n{doc.page_content}")
            chunk_meta.append({
                "citation":   m.get("citation", ""),
                "qa_id":      m.get("qa_id", ""),
                "section":    m.get("section", ""),
                "start_line": m.get("start_line"),
                "end_line":   m.get("end_line"),
            })

        context_str = "\n\n---\n\n".join(context_blocks)

        messages = [
            SystemMessage(
                content=ANSWER_SYSTEM_TMPL.format(context=context_str)
            ),
            HumanMessage(content=standalone_query),
        ]
        response = self._llm.invoke(messages)
        return response.content.strip(), chunk_meta

    # ------------------------------------------------------------------
    # Public: run a full RAG turn
    # ------------------------------------------------------------------
    def ask(self, query: str) -> RAGResponse:
        from langchain_core.messages import HumanMessage, AIMessage

        # 1. Condense
        standalone = self._condense(query)

        # 2. Retrieve
        docs = self._retrieve(standalone)

        # 3. Answer
        answer_text, chunk_meta = self._answer(standalone, docs)

        # 4. Update memory
        self._history.add_message(HumanMessage(content=query))
        self._history.add_message(AIMessage(content=answer_text))

        return RAGResponse(
            query=query,
            rewritten_query=standalone,
            answer=answer_text,
            cited_lines=extract_cited_lines(answer_text),
            source_chunks=chunk_meta,
        )

    def clear_history(self) -> None:
        self._history.clear()
        print("[session] Conversation history cleared.")


# ---------------------------------------------------------------------------
# Factory — load shared resources, return a session factory
# ---------------------------------------------------------------------------
def build_session_factory():
    """
    Loads embeddings, FAISS index, and LLM once.
    Returns a callable  new_session() -> GigaCorpSession.
    """
    from langchain_huggingface            import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq                    import ChatGroq

    index_faiss = VECTORSTORE_DIR / "index.faiss"
    index_pkl = VECTORSTORE_DIR / "index.pkl"

    try:
        if not (index_faiss.exists() and index_pkl.exists()):
            print(
                f"[rag] FAISS index files not found at '{VECTORSTORE_DIR}/'. "
                "Running ingestion automatically ..."
            )
            from ingest import run_ingestion
            run_ingestion(verbose=False)
            print("[rag] Ingestion complete. Continuing with index load ...")
    except FileNotFoundError as e:
        from ingest import FAQ_PATH
        print("=== RAG DIAGNOSTIC INFO ===")
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Script Location: {__file__}")
        print(f"VECTORSTORE_DIR: {VECTORSTORE_DIR} (exists: {VECTORSTORE_DIR.exists()})")
        print(f"FAQ_PATH: {FAQ_PATH} (exists: {FAQ_PATH.exists()})")
        try:
            print(f"Parent directory contents: {os.listdir(Path(__file__).parent.resolve())}")
        except Exception as dir_err:
            print(f"Could not list directory contents: {dir_err}")
        print("===========================")
        raise e

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )

    print(f"[rag] Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[rag] Loading FAISS index from '{VECTORSTORE_DIR}/' ...")
    try:
        db        = FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except FileNotFoundError as e:
        from ingest import FAQ_PATH
        print("=== RAG FAISS LOAD DIAGNOSTIC INFO ===")
        print(f"Current Working Directory: {os.getcwd()}")
        print(f"Script Location: {__file__}")
        print(f"VECTORSTORE_DIR: {VECTORSTORE_DIR} (exists: {VECTORSTORE_DIR.exists()})")
        print(f"FAQ_PATH: {FAQ_PATH} (exists: {FAQ_PATH.exists()})")
        try:
            print(f"Parent directory contents: {os.listdir(Path(__file__).parent.resolve())}")
        except Exception as dir_err:
            print(f"Could not list directory contents: {dir_err}")
        print("===========================")
        raise e

    retriever = db.as_retriever(search_kwargs={"k": TOP_K})

    print(f"[rag] Initialising ChatGroq model: {GROQ_MODEL} ...")
    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=groq_key,
        temperature=0.0,
        max_tokens=1024,
    )

    print("[rag] Ready.\n")

    def new_session() -> GigaCorpSession:
        return GigaCorpSession(retriever=retriever, llm=llm)

    return new_session


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    new_session = build_session_factory()

    # ── Single-shot CLI mode ─────────────────────────────────────────────
    if len(sys.argv) > 1:
        session  = new_session()
        response = session.ask(" ".join(sys.argv[1:]))
        print("=" * 65)
        print(response.pretty())
        print("=" * 65)
        return

    # ── Interactive REPL with persistent session ─────────────────────────
    session = new_session()
    print("GigaCorp FAQ Assistant  (conversational mode)")
    print("Commands: 'reset' = clear history | 'quit' / 'exit' = stop")
    print("=" * 65)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if query.lower() == "reset":
            session.clear_history()
            continue

        response = session.ask(query)

        print()
        if response.rewritten_query != response.query:
            print(f"  [Rewritten] {response.rewritten_query}")
        print(f"  Assistant : {response.answer}")
        if response.cited_lines:
            ranges = ", ".join(f"lines {s}-{e}" for s, e in response.cited_lines)
            print(f"  Sources   : gigacorp_faq.txt [{ranges}]")
        print()


if __name__ == "__main__":
    main()
