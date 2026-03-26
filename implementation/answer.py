import sys
import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
import requests
from dotenv import load_dotenv
from IPython.display import Markdown, display
from openai import OpenAI


load_dotenv(override=True)

#define constants, assigning default values if not set in environment variables
BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434/v1')
MODEL = os.getenv('MODEL', 'llama3.2')
API_KEY = os.getenv('API_KEY', 'ollama')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 5))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

#initialize OpenAI client
openai = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


#def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
def answer_question(question: str,userrole: str, history: list[dict]) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    if userrole != "admin":
        filtered_docs = [
            doc for doc in docs
            if doc.metadata.get("doc_type") in ("company", "products")
        ]
    else:
        filtered_docs = docs
    
    context = "\n\n".join(doc.page_content for doc in filtered_docs)
    # doc.metadata["doc_type"] = doc_type
    system_prompt = SYSTEM_PROMPT.format(context=context)
    #messages = [SystemMessage(content=system_prompt)]
    #messages.extend(convert_to_messages(history))
    #messages.append(HumanMessage(content=question))
    #response = openai.invoke(messages)
    #return response.content, docs
    # Build messages as plain dicts — what openai SDK expects
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    # openai SDK call — works with Ollama's OpenAI-compatible endpoint
    response = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
    )
    return response.choices[0].message.content, docs
