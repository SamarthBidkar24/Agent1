"""
ingest.py
---------
Loads data/gigacorp_faq.txt, splits it into Q&A chunks while preserving
the embedded doc-line numbers as metadata, embeds each chunk with
sentence-transformers/all-MiniLM-L6-v2 (via LangChain HuggingFaceEmbeddings),
and stores the resulting FAISS index under vectorstore/.

The core pipeline is exposed as run_ingestion() so that rag_chain.py
(and any other module) can call it directly to build the index on demand.

Usage (CLI):
    python ingest.py
"""

import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Module-level defaults (used by CLI; callers may override via parameters)
# ---------------------------------------------------------------------------
FAQ_PATH        = Path(__file__).parent.resolve() / "data" / "gigacorp_faq.txt"
VECTORSTORE_DIR = Path(__file__).parent.resolve() / "vectorstore"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 2. Parse the FAQ file into (doc_line_number, text) pairs
# ---------------------------------------------------------------------------

def parse_numbered_lines(filepath: Path) -> list[tuple[int, str]]:
    """
    Each physical line in the file starts with an embedded doc-line number
    (the number we want to cite), then one or more spaces, then the content.

    Example physical line:
        '11 A1.1: GigaCorp offers four shipping tiers ...'
         └─ doc line 11

    Returns a list of (doc_line_no, text_content) tuples, skipping
    blank-content lines (pure whitespace after stripping the number).
    """
    pattern = re.compile(r"^(\d+)\s+(.*)")
    results: list[tuple[int, str]] = []

    with open(filepath, encoding="utf-8") as fh:
        for physical_line in fh:
            m = pattern.match(physical_line)
            if m:
                doc_no  = int(m.group(1))
                content = m.group(2).rstrip()
                results.append((doc_no, content))

    return results


# ---------------------------------------------------------------------------
# 3. Group parsed lines into logical Q&A chunks
# ---------------------------------------------------------------------------

_SECTION_HEADER = re.compile(r"^SECTION\s+\d+:", re.IGNORECASE)
_QA_QUESTION    = re.compile(r"^Q\d+\.\d+:")          # e.g. Q1.1:
_QA_ANSWER      = re.compile(r"^A\d+\.\d+:")          # e.g. A1.1:
_SEPARATOR      = re.compile(r"^-{10,}")               # dashed dividers
_DOC_META       = re.compile(                          # header / footer prose
    r"^(GIGACORP CUSTOMER FAQ|={5,}|Last Updated|Contact:|END OF DOCUMENT"
    r"|GigaCorp, Inc\.|gigacorp\.io)",
    re.IGNORECASE,
)

def build_chunks(parsed_lines: list[tuple[int, str]]) -> list[dict]:
    """
    Walk through the parsed lines and accumulate text into a chunk buffer.
    A new chunk is started whenever a Q-line (question) is encountered.
    The current section name is tracked and injected into every chunk's
    metadata so the retriever knows which section the answer came from.

    Each returned dict has:
        {
            "text":       str,   # full Q+A text (possibly multi-line)
            "start_line": int,   # first doc-line number in this chunk
            "end_line":   int,   # last  doc-line number in this chunk
            "section":    str,   # e.g. "SECTION 1: SHIPPING POLICIES"
            "qa_id":      str,   # e.g. "Q1.2" (empty for non-Q&A chunks)
        }
    """
    chunks: list[dict] = []
    current_section = "PREAMBLE"
    current_chunk: list[tuple[int, str]] = []
    current_qa_id  = ""

    def flush(chunk_lines: list[tuple[int, str]], qa_id: str) -> None:
        if not chunk_lines:
            return
        text = "\n".join(c for _, c in chunk_lines).strip()
        if not text:
            return
        chunks.append({
            "text":       text,
            "start_line": chunk_lines[0][0],
            "end_line":   chunk_lines[-1][0],
            "section":    current_section,
            "qa_id":      qa_id,
        })

    for doc_no, content in parsed_lines:
        if not content or _SEPARATOR.match(content):
            continue
        if _SECTION_HEADER.match(content):
            flush(current_chunk, current_qa_id)
            current_chunk  = []
            current_qa_id  = ""
            current_section = content.strip()
            continue
        if _DOC_META.match(content):
            continue
        if _QA_QUESTION.match(content):
            flush(current_chunk, current_qa_id)
            current_chunk = []
            m = re.match(r"^(Q\d+\.\d+):", content)
            current_qa_id = m.group(1) if m else ""
        current_chunk.append((doc_no, content))

    flush(current_chunk, current_qa_id)
    return chunks


# ---------------------------------------------------------------------------
# 4. Convert chunks to LangChain Documents
# ---------------------------------------------------------------------------

def chunks_to_documents(chunks: list[dict], faq_path: Path = FAQ_PATH):
    """Wrap each chunk dict in a LangChain Document with rich metadata."""
    from langchain_core.documents import Document

    docs = []
    for chunk in chunks:
        metadata = {
            "source":     str(faq_path),
            "section":    chunk["section"],
            "qa_id":      chunk["qa_id"],
            "start_line": chunk["start_line"],
            "end_line":   chunk["end_line"],
            "citation": (
                f"gigacorp_faq.txt lines "
                f"{chunk['start_line']}-{chunk['end_line']}"
            ),
        }
        docs.append(Document(page_content=chunk["text"], metadata=metadata))

    return docs


# ---------------------------------------------------------------------------
# 5. Embed + persist FAISS index
# ---------------------------------------------------------------------------

def build_and_save_index(
    docs,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    embed_model: str = EMBED_MODEL,
) -> int:
    """Embed documents, save FAISS index, return number of docs stored."""
    from langchain_huggingface            import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print(f"[ingest] Loading embedding model: {embed_model} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[ingest] Embedding {len(docs)} chunks ...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(vectorstore_dir))
    print(f"[ingest] FAISS index saved to '{vectorstore_dir}/'")

    return len(docs)


# ---------------------------------------------------------------------------
# 6. Public pipeline function — importable by rag_chain.py and others
# ---------------------------------------------------------------------------

def run_ingestion(
    faq_path: Path = FAQ_PATH,
    vectorstore_dir: Path = VECTORSTORE_DIR,
    embed_model: str = EMBED_MODEL,
    verbose: bool = True,
) -> int:
    """
    Full ingestion pipeline: parse → chunk → embed → save FAISS index.

    Parameters
    ----------
    faq_path        : Path to the numbered FAQ text file.
    vectorstore_dir : Directory where the FAISS index will be saved.
    embed_model     : HuggingFace model name for sentence embeddings.
    verbose         : If True, print chunk-level debug info to stdout.

    Returns
    -------
    int : Number of chunks indexed.

    This function is safe to import and call from other modules.
    rag_chain.py calls it automatically when the vectorstore is missing.
    """
    if not faq_path.exists():
        raise FileNotFoundError(
            f"[ingest] FAQ source file not found: {faq_path}\n"
            f"Make sure '{faq_path}' exists before running ingestion."
        )

    print(f"[ingest] Parsing '{faq_path}' ...")
    parsed_lines = parse_numbered_lines(faq_path)
    print(f"[ingest] Read {len(parsed_lines)} numbered lines.")

    print("[ingest] Building Q&A chunks ...")
    chunks = build_chunks(parsed_lines)

    if verbose:
        for i, c in enumerate(chunks, 1):
            print(
                f"  Chunk {i:>2}: [{c['qa_id'] or 'narrative':>6}] "
                f"lines {c['start_line']:>3}-{c['end_line']:>3}  "
                f"| {c['section']}"
            )

    docs  = chunks_to_documents(chunks, faq_path)
    count = build_and_save_index(docs, vectorstore_dir, embed_model)

    print(f"[ingest] Done! {count} chunks indexed and saved to '{vectorstore_dir}/'.")
    return count


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    run_ingestion(
        faq_path=FAQ_PATH,
        vectorstore_dir=VECTORSTORE_DIR,
        embed_model=EMBED_MODEL,
        verbose=True,
    )


if __name__ == "__main__":
    main()
