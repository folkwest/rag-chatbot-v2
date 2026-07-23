import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.title("📄 RAG Chatbot with Strategy Comparison")

uploaded = st.file_uploader("Upload a document", type=["pdf", "txt"])

# Multi-select for chunking strategies
chunking = st.multiselect(
    "Chunking strategies to compare",
    ["fixed", "sentence", "semantic"],
    default=["fixed"],
)

# Retrieval strategy selector
retrieval = st.selectbox(
    "Retrieval strategy",
    ["vector", "hybrid"],
    index=0,
    help="'vector' uses cosine similarity only. 'hybrid' adds BM25 keyword search + cross-encoder reranking.",
)

question = st.text_input("Ask a question")

# Use session state to persist doc_id across reruns
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

if uploaded:
    # Only upload once per file
    file_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("uploaded_file_key") != file_key:
        with st.spinner("Uploading and processing document..."):
            try:
                upload_resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=300,
                )
                upload_resp.raise_for_status()
                data = upload_resp.json()
                st.session_state.doc_id = data["document_id"]
                st.session_state.uploaded_file_key = file_key
                st.success(
                    f"Uploaded **{data['filename']}** — "
                    f"chunks per strategy: {data['chunks_per_strategy']}"
                )
            except requests.exceptions.RequestException as e:
                st.error(f"Upload failed: {e}")

if st.button("Ask") and question and st.session_state.doc_id and chunking:
    with st.spinner("Generating answer..."):
        try:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={
                    "question": question,
                    "document_id": st.session_state.doc_id,
                    "chunking_strategy": chunking,
                    "retrieval_strategy": retrieval,
                },
                timeout=60,
            )
            resp.raise_for_status()
            chat_resp = resp.json()

            for result in chat_resp["results"]:
                st.subheader(f"Strategy: {result['strategy']}")
                st.caption(f"Retrieval: {retrieval}")
                st.write("**Answer:**", result["answer"])
                st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")

                with st.expander(f"Sources ({len(result['sources'])} chunks)"):
                    for src in result["sources"]:
                        score_text = f"score: {src['score']:.4f}"
                        if src.get("reranker_score") is not None:
                            score_text += f" | reranker: {src['reranker_score']:.4f}"
                        st.caption(f"{src['filename']} [{src['chunking_strategy']}] — {score_text}")
                        st.write(src["text"][:500])
                        st.divider()
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
elif st.button("Ask") if False else False:
    pass  # placeholder to avoid streamlit warnings
