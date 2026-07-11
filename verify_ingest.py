"""
verify_ingest.py
----------------
Loads the FAISS index from vectorstore/, runs a similarity search for
"What is your return policy?", and prints the top 3 results with their
citation strings.

Usage:
    python verify_ingest.py
"""

import os
import warnings
from pathlib import Path

# Suppress noisy deprecation warnings for a clean demo output
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

VECTORSTORE_DIR = Path(__file__).parent.resolve() / "vectorstore"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
QUERY           = "What is your return policy?"
TOP_K           = 3


def main() -> None:
    from langchain_huggingface          import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"No vectorstore found at '{VECTORSTORE_DIR}/'. "
            "Run ingest.py first."
        )

    # --- Load embeddings (must match what was used during ingestion) ---
    print(f"Loading embedding model: {EMBED_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # --- Load FAISS index ---
    print(f"Loading FAISS index from '{VECTORSTORE_DIR}/' ...")
    db = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,   # safe: index was built locally
    )

    # --- Similarity search ---
    print(f'\nQuery: "{QUERY}"\n')
    print("=" * 65)

    results = db.similarity_search_with_score(QUERY, k=TOP_K)

    for rank, (doc, score) in enumerate(results, start=1):
        meta = doc.metadata
        print(f"Result #{rank}")
        print(f"  Citation  : {meta.get('citation', 'N/A')}")
        print(f"  Section   : {meta.get('section', 'N/A')}")
        print(f"  QA ID     : {meta.get('qa_id', 'N/A')}")
        print(f"  Lines     : {meta.get('start_line')} - {meta.get('end_line')}")
        print(f"  Score     : {score:.4f}  (lower = more similar for L2)")
        print(f"  Content   :")
        # Indent the chunk text for readability
        for line in doc.page_content.splitlines():
            print(f"    {line}")
        print("-" * 65)

    print(f"\nTop {TOP_K} chunks retrieved successfully.")


if __name__ == "__main__":
    main()
