import time
import warnings
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", timeout=25, max_retries=0)


def ask_llm(prompt, fallback: str = ""):
    for attempt in range(3):
        try:
            content = llm.invoke(prompt).content
            if isinstance(content, list):
                content = "".join(p["text"] for p in content if isinstance(p, dict) and "text" in p)
            return content.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  [API Unavailable]: {e}")
                return fallback
            time.sleep(2 * (attempt + 1))


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
)

vector_store = Chroma(
    collection_name="chunks",
    embedding_function=embeddings,
    persist_directory="vectorstore",
)

stored_data = vector_store.get()
bm25_retriever = BM25Retriever.from_texts(
    texts=stored_data["documents"],
    metadatas=stored_data["metadatas"],
    k=2,
)


class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    retry_count: int
    relevant_threshold: int
    chunks: list
    is_relevant: bool


def check_relevance(state: State):
    sys_prompt = (
        "Determine if the user's latest question is related to AI, machine learning, "
        "generative AI risks, governance, security, or NIST frameworks—OR is a follow-up "
        "to the immediately preceding AI-related exchange. Reply ONLY with 'YES' or 'NO'."
    )
    prompt = [
        SystemMessage(content=sys_prompt),
        *state.get("messages", [])[-2:],
        HumanMessage(content=state["query"]),
    ]
    verdict = ask_llm(prompt, fallback="YES").upper()
    return {"is_relevant": "YES" in verdict}


def re_write_query(state: State):
    if state["retry_count"] == 0:
        sys_prompt = (
            "Rewrite the user's question so its vocabulary matches NIST AI RMF documents. "
            "If it is a follow-up question, resolve pronouns using chat history while "
            "preserving the exact aspect asked about (such as AI actors, risks, or actions). "
            "Return only the rewritten query."
        )
    else:
        sys_prompt = (
            "The previous query did not retrieve relevant chunks. "
            "Regenerate it using concise NIST AI RMF terms or a brief hypothetical answer (HyDE). "
            "Return only the improved query."
        )

    prompt = [
        SystemMessage(content=sys_prompt),
        *state.get("messages", [])[-2:],
        HumanMessage(content=state["query"]),
    ]
    return {"query": ask_llm(prompt, fallback=state["query"])}


def retrieve_chunks(state: State):
    query = state["query"]
    semantic_docs = vector_store.similarity_search(query, k=2)
    bm25_docs = bm25_retriever.invoke(query)

    seen = set()
    combined = []
    for doc in semantic_docs + bm25_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    return {"chunks": combined}


def grade_and_filter(state: State):
    chunks = state["chunks"]
    numbered_chunks = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(chunks))

    sys_prompt = (
        "You are a relevance grader. Given a query and numbered document chunks, "
        "reply ONLY with the comma-separated numbers of the chunks that help answer the query "
        "(e.g. '0, 2'), or reply 'NONE' if none are relevant."
    )
    prompt = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Query: {state['query']}\n\nChunks:\n{numbered_chunks}"),
    ]
    verdict = ask_llm(prompt, fallback="NONE").upper()

    useful_chunks = (
        [] if "NONE" in verdict else [doc for i, doc in enumerate(chunks) if str(i) in verdict]
    )

    if len(useful_chunks) < state["relevant_threshold"]:
        return {"chunks": useful_chunks, "retry_count": state["retry_count"] + 1}

    return {"chunks": useful_chunks}


def generate_answer(state: State):
    context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('source')} | Page {doc.metadata.get('page')}]\n{doc.page_content}"
        for doc in state["chunks"]
    )

    sys_prompt = (
        "Answer the user's question ONLY using the provided NIST document chunks below. "
        "Do not use outside knowledge. Cite the source document name and page number for every claim.\n\n"
        f"Context:\n{context}"
    )
    prompt = [
        SystemMessage(content=sys_prompt),
        *state.get("messages", [])[-2:],
        HumanMessage(content=state["query"]),
    ]
    answer = ask_llm(prompt, fallback="Sorry, the model timed out. Please try again.")

    return {
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=answer)],
        "retry_count": 0,
    }


def fallback_refusal(state: State):
    refusal = (
        "I cannot answer this question because I could not find relevant information "
        "in the NIST AI RMF 1.0 or Generative AI Profile documents."
    )
    return {
        "messages": [HumanMessage(content=state["query"]), AIMessage(content=refusal)],
        "retry_count": 0,
    }


def route_initial(state: State) -> str:
    return "re_write_query" if state["is_relevant"] else "fallback_refusal"


def route_after_grading(state: State) -> str:
    if len(state["chunks"]) >= state["relevant_threshold"]:
        return "generate_answer"
    return "re_write_query" if state["retry_count"] <= 1 else "fallback_refusal"


builder = StateGraph(State)

builder.add_node("check_relevance", check_relevance)
builder.add_node("re_write_query", re_write_query)
builder.add_node("retrieve_chunks", retrieve_chunks)
builder.add_node("grade_and_filter", grade_and_filter)
builder.add_node("generate_answer", generate_answer)
builder.add_node("fallback_refusal", fallback_refusal)

builder.add_edge(START, "check_relevance")
builder.add_conditional_edges("check_relevance", route_initial)
builder.add_edge("re_write_query", "retrieve_chunks")
builder.add_edge("retrieve_chunks", "grade_and_filter")
builder.add_conditional_edges("grade_and_filter", route_after_grading)
builder.add_edge("generate_answer", END)
builder.add_edge("fallback_refusal", END)

graph = builder.compile(checkpointer=InMemorySaver())