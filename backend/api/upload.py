import os
import tempfile
import uuid
import logging

from fastapi import APIRouter, UploadFile, HTTPException
from backend.chunking import get_chunker
from backend.utils.parsing import parse_pdf, parse_txt
from backend.utils.embeddings import embed_texts
from backend.vectorstore.chroma_store import vector_store
from backend.storage.document_store import doc_store
from backend.retrieval.bm25_registry import bm25_registry

logger = logging.getLogger(__name__)

router = APIRouter()

# All strategies to pre-compute during upload
ALL_STRATEGIES = ["fixed", "sentence", "semantic"]


@router.post("/upload")
async def upload(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    doc_id = str(uuid.uuid4())

    # Write to a secure temp file
    suffix = os.path.splitext(file.filename)[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        tmp.write(content)
        tmp.close()

        # Parse based on file type
        if file.filename.lower().endswith(".pdf"):
            text = parse_pdf(tmp.name)
        else:
            text = parse_txt(tmp.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    strategy_info = {}

    for strategy in ALL_STRATEGIES:
        chunker = get_chunker(strategy)
        chunks = chunker.chunk(text)

        if not chunks:
            logger.warning(f"Strategy '{strategy}' produced 0 chunks for {file.filename}")
            strategy_info[strategy] = 0
            continue

        embeddings = embed_texts(chunks)

        metadatas = [
            {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunk_id": i,
                "chunk_index": i,
                "chunking_strategy": strategy,
                "source_filename": file.filename,
            }
            for i in range(len(chunks))
        ]

        # Store in the strategy-specific ChromaDB collection
        vector_store.add(
            embeddings=embeddings,
            texts=chunks,
            metadatas=metadatas,
            strategy=strategy,
        )

        # Index chunks with BM25 for hybrid retrieval
        chunk_ids = [f"{doc_id}__{strategy}__{i}" for i in range(len(chunks))]
        bm25_registry.add_documents(strategy, doc_id, chunks, chunk_ids)

        strategy_info[strategy] = len(chunks)

    # Register document in metadata store
    doc_store.add(doc_id, file.filename)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "chunks_per_strategy": strategy_info,
    }
