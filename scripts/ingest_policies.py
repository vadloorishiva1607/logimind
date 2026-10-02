"""
RAG ingestion script -- ChromaDB version.

Run by an admin, NOT by the chatbot or any agent -- same principle as the
pgvector version: write access to the knowledge base is a separate,
manually-triggered action, never something a chat request can do.

Storage: Chroma runs in EMBEDDED mode -- no separate server process, just
a persistent directory on disk (CHROMA_PERSIST_DIR). This keeps the EC2
instance's footprint to one process instead of two, which matters on a
1GB t3.micro.

Read-access control: Chroma metadata values must be scalars (no native
array-contains filter), so instead of one visible_to_roles array column
(the pgvector approach), each role gets its own boolean metadata field
per chunk -- e.g. role_finance=True. The RAG agent then filters with
where={"role_<user_role>": True} at query time.
"""
import os

import chromadb
from chromadb.utils import embedding_functions

from app.llm import get_embeddings

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
COLLECTION_NAME = "policy_chunks"

POLICIES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policies")

ALL_ROLES = ["ops_manager", "dispatcher", "finance", "driver_manager", "compliance"]

DOC_VISIBILITY = {
    "driver_termination_policy.txt": ["driver_manager", "compliance", "ops_manager"],
    "fuel_reimbursement_policy.txt": ALL_ROLES,
    "escrow_withdrawal_policy.txt": ["finance", "compliance", "ops_manager"],
    "settlement_calculation_sop.txt": ["finance", "ops_manager", "compliance"],
    "driver_onboarding_sop.txt": ["driver_manager", "ops_manager"],
    "safety_performance_review_policy.txt": ["driver_manager", "ops_manager", "compliance"],
    "shipment_delay_risk_sop.txt": ["dispatcher", "ops_manager"],
    "confidential_data_handling_policy.txt": ALL_ROLES,
}


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    """Simple paragraph-aware chunking -- these policy docs are short
    enough that most become 1-2 chunks each."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += p + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def role_metadata(roles: list[str]) -> dict:
    """One boolean field per role -- e.g. {'role_finance': True,
    'role_compliance': True} for the roles allowed to see this chunk.
    Roles NOT in the list simply don't get a field (Chroma's `where`
    filter only needs to check the field for the requesting role)."""
    return {f"role_{r}": True for r in roles}


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    # embedding_function=None because we compute and pass embeddings
    # ourselves via app.llm.get_embeddings() -- keeps the SAME OpenAI
    # embedding model used everywhere else in the app, rather than
    # letting Chroma default to its own local model.
    return client.get_or_create_collection(name=COLLECTION_NAME)


def ingest():
    embeddings = get_embeddings()
    collection = get_chroma_collection()

    total_chunks = 0
    for filename, roles in DOC_VISIBILITY.items():
        path = os.path.join(POLICIES_DIR, filename)
        with open(path) as f:
            text = f.read()

        doc_title = filename.replace("_", " ").replace(".txt", "").title()
        chunks = chunk_text(text)

        ids, vectors, documents, metadatas = [], [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{filename}::chunk{i}"
            vector = embeddings.embed_query(chunk)
            ids.append(chunk_id)
            vectors.append(vector)
            documents.append(chunk)
            metadatas.append({
                "doc_title": doc_title,
                "doc_source": filename,
                **role_metadata(roles),
            })

        # upsert = safe to re-run: existing chunk IDs get overwritten,
        # not duplicated, if you re-ingest after editing a document.
        collection.upsert(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
        total_chunks += len(chunks)
        print(f"Ingested {filename}: {len(chunks)} chunk(s), visible to {roles}")

    print(f"\nDone. {total_chunks} total chunks in Chroma collection '{COLLECTION_NAME}' at {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    ingest()
