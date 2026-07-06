from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_pinecone import PineconeVectorStore

# Reuse YOUR existing helper functions (same ones store_index.py uses)
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)


# ----------------------------------------------------------------------
# Build the same chunks store_index.py built (needed for in-memory BM25).
# ----------------------------------------------------------------------
def _get_chunks(data_dir: str = "data/"):
    extracted = load_pdf_file(data_dir)
    minimal = filter_to_minimal_docs(extracted)
    return text_split(minimal)


# ----------------------------------------------------------------------
# Hybrid + re-ranked retriever.
# ----------------------------------------------------------------------
def build_hybrid_retriever(
    index_name: str,
    documents=None,
    data_dir: str = "data/",
    embeddings=None,
    k: int = 20,           
    top_n: int = 4,         
    dense_weight: float = 0.6,
    sparse_weight: float = 0.4,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
):
    """
    Drop-in replacement for `docsearch.as_retriever(...)`.
    Returns a retriever you plug straight into your existing chain.
    """
    # Same HuggingFace embeddings you indexed with
    if embeddings is None:
        embeddings = download_hugging_face_embeddings()

    # --- Dense: your existing Pinecone index (NO re-indexing needed) ---
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # --- Sparse: BM25 in memory over the same chunks ---
    if documents is None:
        documents = _get_chunks(data_dir)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # --- Combine (weighted Reciprocal Rank Fusion) ---
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[sparse_weight, dense_weight],
    )

    # --- Re-rank fused candidates with a cross-encoder ---
    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_n)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble,
    )
