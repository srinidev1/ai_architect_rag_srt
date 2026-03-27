# Insurellm RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline that lets users query an insurance company knowledge base through a conversational chat interface. The project is built on top of a dataset from the **Ed Donner tutorial** and is presented as a multi-user Streamlit web application.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Ollama — `llama3.2` (local) |
| Embedding model | HuggingFace — `all-MiniLM-L6-v2` |
| Vector store | ChromaDB |
| Evaluation judge | EPAM DIAL API |
| Frontend | Streamlit |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with the `llama3.2` model pulled
- A valid EPAM DIAL API key (required for evaluation)

### Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root. All configuration values have sensible defaults **except** `DIAL_API_KEY`, which is mandatory.

```env
# Required
DIAL_API_KEY=your_dial_api_key_here

# Optional — defaults shown below
BASE_URL=http://localhost:11434/v1
MODEL=llama3.2
API_KEY=ollama
RETRIEVAL_K=10
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Running the Application

```bash
streamlit run main.py
```

---

## Authentication

The application uses basic credential-based authentication. Login credentials are stored in `config.yaml`. The features available after login depend on the user's assigned role.

> **Note:** Role-based access here refers to UI feature visibility, not to access-control-aware RAG context filtering (which is described separately below).

---

## User Roles & Features

### Admin

#### 📥 Ingest
- Supports both **initial** and **incremental** ingestion of documents into the vector store.
- Incremental ingestion enables **corpus updates without a full Vector DB rebuild**, avoiding costly re-indexing of unchanged documents.
- Enabling **Show 3-D Vector Visualisation** before clicking *Run Ingest* generates an interactive 3-D plot using **t-SNE** dimensionality reduction.

#### 💬 Chat
- All retrieved context documents (up to the configured `RETRIEVAL_K` limit) are passed to the LLM when generating a response.
- No document-type filtering is applied — admin users receive full context including employee and contract documents.

#### 📊 Evaluation
- **Retrieval Evaluation** — measures retrieval quality using MRR, nDCG, and keyword coverage metrics.
- **Answer Evaluation** — scores LLM responses on accuracy, completeness, and relevance using the EPAM DIAL API as the judge.

Both evaluation modes support an optional **reranking** step:

- When the **Enable Reranking** option is selected, the retriever initially fetches **2× the configured `RETRIEVAL_K`** candidate documents from ChromaDB via Ollama.
- Those candidates are then sent to the **EPAM DIAL API** for reranking, which scores and reorders them by relevance to the query.
- Only the **top K documents** from the reranked results are ultimately passed to the LLM for answer generation.

This two-stage retrieval approach improves context quality by using a more capable model to refine the initial vector-similarity results before they influence the final answer.

---

### Customer

#### 💬 Chat (access-control-aware)
Customer users can chat with the assistant, but with **context-level access control** applied:

- Only documents of type `company` or `products` are included in the context passed to the LLM.
- Documents of type `employees` and `contracts` are **excluded** from the context.

**Example:** If a user asks *"Who is Avery?"*, an admin will receive a full answer because Avery's details exist in an employee document. A customer user will not, because that document type is filtered out before the LLM is called.

> This implements **access-control-aware RAG** at the context-retrieval layer — not at the UI level.


## Notes

- The dataset used in this project is sourced from the **Ed Donner RAG tutorial**.
- The EPAM DIAL API is used exclusively as a judge model during evaluation and reranking — it is not involved in the standard chat pipeline.
- Ollama must be running locally before starting the application.

## Changelog

| Date | Change |
|---|---|
| 2026-03-27 | Added reranking support to Retrieval and Answer evaluations — fetches 2× K candidates via Ollama and reranks using EPAM DIAL API before selecting top K for the final answer. |
