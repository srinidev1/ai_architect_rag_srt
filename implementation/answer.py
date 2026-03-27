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
from pydantic import BaseModel, Field
from openai import AzureOpenAI

load_dotenv(override=True)

#define constants, assigning default values if not set in environment variables
BASE_URL = os.getenv('BASE_URL', 'http://localhost:11434/v1')
MODEL = os.getenv('MODEL', 'llama3.2')
API_KEY = os.getenv('API_KEY', 'ollama')
RETRIEVAL_K = int(os.getenv('RETRIEVAL_K', 5))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

dial_api_key = os.getenv('DIAL_API_KEY')
DIALMODEL =  os.getenv('AZURE_MODEL', "gpt-4")

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

dialclient = AzureOpenAI(
    api_key         = dial_api_key,
    api_version     = "2024-08-01-preview",
    azure_endpoint  = "https://ai-proxy.lab.epam.com",
)

class Result(BaseModel):
    page_content: str
    metadata: dict


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )

def fetch_context(question: str, use_rerank: bool = False) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    if rerank:
        # If reranking is enabled, retrieve more documents and then filter them
        initial_docs = retriever.invoke(question, k=RETRIEVAL_K * 2)
        # Simple reranking: sort by relevance score (if available) or keep original order
        sorted_docs = rerank(question, initial_docs)
        #sorted(initial_docs, key=lambda d: d.metadata.get("relevance_score", 0), reverse=True)
        return sorted_docs[:RETRIEVAL_K]
    elif not rerank:
        return retriever.invoke(question, k=RETRIEVAL_K)

def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = dialclient.chat.completions.parse(model=DIALMODEL,messages=messages,response_format=RankOrder)
    reply = response.choices[0].message.content
    order = RankOrder.model_validate_json(reply).order
    return [chunks[i - 1] for i in order]


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


#def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
def answer_question(question: str,userrole: str,rerank: bool, history: list[dict]) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined, use_rerank=rerank)
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
