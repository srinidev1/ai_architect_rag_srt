import os
import glob
import hashlib
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL          = "gpt-4.1-nano"
DB_NAME        = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")

load_dotenv(override=True)
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ── Shared helpers ─────────────────────────────────────────────────────────────

def fetch_documents(foldername :str):
    """Load all .md files from every sub-folder of KNOWLEDGE_BASE.
    Each document gets a `doc_type` metadata key equal to its parent folder name.
    """
    target_dir = Path(KNOWLEDGE_BASE) / foldername
    folders    = glob.glob(str(target_dir / "*"))
    documents  = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader   = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_documents(documents)


def _chunk_hash(chunk) -> str:
    """Stable SHA-256 fingerprint for a chunk (source path + content)."""
    raw = f"{chunk.metadata.get('source', '')}::{chunk.page_content}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _collection_stats(vectorstore) -> tuple[int, int]:
    """Return (count, dimensions) for a Chroma vectorstore."""
    collection      = vectorstore._collection
    count           = collection.count()
    sample          = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions      = len(sample)
    return count, dimensions


# ── Public API ─────────────────────────────────────────────────────────────────

def _initial_ingest() -> dict:
    """
    Drop the existing Chroma collection (if any), re-embed every chunk from
    scratch, and persist the new collection.

    Returns
    -------
    dict with keys:
        docs_processed  – number of source documents loaded
        chunks_upserted – number of chunks written to the store
        chunks_skipped  – always 0 for a full rebuild
        collection_name – Chroma collection name (directory basename)
        vector_details  – total number of vectors in the collection after ingest with dimensions info for optional 3-D visualisation
        collection      – Chroma collection object for optional 3-D visualisation
    """
    documents = fetch_documents("initial")
    chunks    = create_chunks(documents)

    # Wipe existing store
    if os.path.exists(DB_NAME):
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    count, dimensions = _collection_stats(vectorstore)
    print(
        f"[initial_ingest] {count:,} vectors with {dimensions:,} dimensions "
        f"written to '{DB_NAME}'"
    )

    return {
        "docs_processed"  : len(documents),
        "chunks_upserted" : len(chunks),
        "chunks_skipped"  : 0,
        "collection_name" : os.path.basename(DB_NAME),
        "vector_details"  : f"{count:,} vectors with {dimensions:,} dimensions",
        "collection"      : vectorstore._collection  # return collection for optional 3-D visualisation 
    }


def _incremental_ingest() -> dict:
    """
    Compare incoming chunks against what is already stored (using a SHA-256
    hash of source path + content stored in chunk metadata).  Only new or
    changed chunks are upserted; untouched chunks are left as-is.

    Returns
    -------
    dict with keys:
        docs_processed  – number of source documents loaded
        chunks_upserted – number of chunks added / replaced
        chunks_skipped  – number of chunks that were already up-to-date
        collection_name – Chroma collection name (directory basename)
        vector_details  – total number of vectors in the collection after ingest with dimensions info for optional 3-D visualisation
        collection      – Chroma collection object for optional 3-D visualisation
    """
    documents = fetch_documents("incremental")
    chunks    = create_chunks(documents)

    # Open (or create) the existing store without wiping it
    vectorstore = Chroma(
        persist_directory=DB_NAME,
        embedding_function=embeddings,
    )
    collection = vectorstore._collection

    # Build a set of hashes already present in the store
    existing_records = collection.get(include=["metadatas"])
    existing_hashes  = {
        meta.get("chunk_hash")
        for meta in existing_records["metadatas"]
        if meta and meta.get("chunk_hash")
    }

    new_chunks = []
    skipped    = 0

    for chunk in chunks:
        h = _chunk_hash(chunk)
        if h in existing_hashes:
            skipped += 1
        else:
            chunk.metadata["chunk_hash"] = h   # store hash for future runs
            new_chunks.append(chunk)

    if new_chunks:
        vectorstore.add_documents(new_chunks)

    count, dimensions = _collection_stats(vectorstore)
    print(
        f"[incremental_ingest] upserted {len(new_chunks):,} / skipped {skipped:,} chunks — "
        f"store now has {count:,} vectors with {dimensions:,} dimensions"
    )

    return {
        "docs_processed"  : len(documents),
        "chunks_upserted" : len(new_chunks),
        "chunks_skipped"  : skipped,
        "collection_name" : os.path.basename(DB_NAME),
        "vector_details"  : f"{count:,} vectors with {dimensions:,} dimensions",
        "collection"      : vectorstore._collection  # return collection for optional 3-D visualisation 
    }
