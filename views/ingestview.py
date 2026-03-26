from unittest import result

from implementation.ingest import _incremental_ingest, _initial_ingest
import streamlit as st
import numpy as np
from router import require_role
from langchain_chroma import Chroma
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def render():
    require_role(["admin"])
    showGraph = False
    st.title("📥 Ingest")
    st.write("Upload and process data into the vector datastore.")

    # ── Ingest mode selection ──────────────────────────────────────────────────
    ingest_mode = st.radio(
        "Select Ingest Mode",
        options=["Initial Ingest", "Incremental Ingest"],
        help=(
            "**Initial Ingest** – Loads all documents from scratch and rebuilds the entire Chroma collection.\n\n"
            "**Incremental Ingest** – Only adds or updates new/changed document chunks; existing unchanged chunks are preserved."
        ),
    )

    st.markdown("---")

    if ingest_mode == "Initial Ingest":
        st.info(
            "⚠️ **Initial Ingest** will drop and recreate the Chroma collection. "
            "All existing vectors will be replaced."
        )
    else:
        st.info(
            "🔄 **Incremental Ingest** will compare document hashes and upsert only "
            "new or modified chunks. Existing unchanged chunks are kept as-is."
        )

    st.markdown("---")
    if st.checkbox("🧊 Show 3-D Vector Store Visualisation", value=False):
        showGraph = True

    # ── Submit ─────────────────────────────────────────────────────────────────
    if st.button("▶ Run Ingest", type="primary", use_container_width=True):
        run_ingest(ingest_mode,showGraph)


# ── Core ingest logic ──────────────────────────────────────────────────────────

def run_ingest(mode: str, showGraph: bool = False):
    """Execute the chosen ingest mode and surface result + optional 3-D plot."""
    status_placeholder = st.empty()

    try:
        with st.spinner(f"Running {mode} …"):
            if mode == "Initial Ingest":
                result = _initial_ingest()
            else:
                result = _incremental_ingest()

        # ── Success banner ─────────────────────────────────────────────────────
        status_placeholder.success(
            f"✅ **{mode} completed successfully!**\n\n"
            f"- Documents processed : {result['docs_processed']}\n"
            f"- Chunks upserted     : {result['chunks_upserted']}\n"
            f"- Chunks skipped      : {result['chunks_skipped']}\n"
            f"- Collection          : `{result['collection_name']}`\n"
            f"- Store now has      : {result['vector_details']}"
        )
        print(f"showGraph: {showGraph}")
        # ── Optional 3-D vector visualisation ─────────────────────────────────
        if showGraph:
            collection = result.get("collection")
            render_3d_vectors(collection)
        
       

    except Exception as exc:  # noqa: BLE001
        status_placeholder.error(
            f"❌ **{mode} failed.**\n\n"
            f"**Error:** `{exc}`\n\n"
            "Please check the logs for details and try again."
        )
        st.exception(exc)  # full traceback in an expander





# ── 3-D visualisation ──────────────────────────────────────────────────────────

def render_3d_vectors(collection):
    """
    Pull all vectors + metadata from ChromaDB, reduce to 3-D with t-SNE,
    and render an interactive Plotly scatter plot inside Streamlit.
    """
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vectors = np.array(result['embeddings'])
    documents = result['documents']
    metadatas = result['metadatas']
    doc_types = [metadata['doc_type'] for metadata in metadatas]
    colors = [['blue', 'green', 'red', 'orange'][['products', 'employees', 'contracts', 'company'].index(t)] for t in doc_types]
   
    tsne = TSNE(n_components=3, random_state=42)
    reduced_vectors = tsne.fit_transform(vectors)
    # Create the 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    z=reduced_vectors[:, 2],
    mode='markers',
    marker=dict(size=5, color=colors, opacity=0.8),
    text=[f"Type: {t}<br>Text: {d[:100]}..." for t, d in zip(doc_types, documents)],
    hoverinfo='text'
    )])

    fig.update_layout(
    title='3D Chroma Vector Store Visualization',
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='z'),
    width=900,
    height=700,
    margin=dict(r=10, b=10, l=10, t=40)
    )
    st.title("3D Vector Visualization")
    st.plotly_chart(fig, use_container_width=True)