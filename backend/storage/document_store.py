from typing import Dict, Optional


class DocumentStore:
    """Simple in-memory document registry."""

    def __init__(self):
        self.documents: Dict[str, dict] = {}

    def add(self, doc_id: str, filename: str):
        self.documents[doc_id] = {"filename": filename}

    def get(self, doc_id: str) -> Optional[dict]:
        return self.documents.get(doc_id)

    def exists(self, doc_id: str) -> bool:
        return doc_id in self.documents

    def list_docs(self) -> Dict[str, dict]:
        return self.documents


doc_store = DocumentStore()
