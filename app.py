import os
import shutil
import uuid
from dotenv import load_dotenv
from typing import List, Annotated
import time
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Assuming you might want PDF back
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

import json
import argparse


# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()  # Load environment variables from .env file
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Define directories
PROFILE_DIR = "./json"  # Changed from ngo_profiles to json
DB_DIR = "./chroma_db"

# --- 2. PHASE 1: KNOWLEDGE BASE CONSTRUCTION ---


def build_rag_pipeline():
    """
    Builds the RAG pipeline by loading, splitting, and indexing documents.
    Crucially, it loads from an existing DB if available to avoid re-embedding.
    Returns a retriever object.
    """
    use_mock = os.getenv("USE_MOCK_EMBEDDINGS", "false").lower() == "true"

    if use_mock:
        print("Using mock embeddings for testing...")
        embeddings = MockEmbeddings()
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001")

    # --- KEY CHANGE: Check for and load an existing database ---
    if os.path.exists(DB_DIR) and any(os.scandir(DB_DIR)):
        print(f"Loading existing vector database from '{DB_DIR}'...")
        vector_store = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        print("Database loaded successfully.")
        return vector_store.as_retriever()

    # --- This section runs ONLY if the database does not exist ---
    print(
        f"No existing database found. Building a new one from documents in '{PROFILE_DIR}'...")
    if not os.path.exists(PROFILE_DIR):
        print(
            f"Error: Profile directory '{PROFILE_DIR}' not found. Please create it and add your JSON files.")
        return None

    # Load JSON files
    documents = []
    for filename in os.listdir(PROFILE_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(PROFILE_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert json content to a string for the document
                    page_content = json.dumps(data, indent=2)
                    documents.append(
                        Document(page_content=page_content, metadata={"source": filename}))
            except Exception as e:
                print(
                    f"Warning: Could not read or process {filename}. Error: {e}")

    if not documents:
        print(f"\n--- ERROR ---")
        print(
            f"No valid JSON documents were found in the '{PROFILE_DIR}' directory.")
        print(f"---------------")
        exit()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(
        f"Created {len(chunks)} document chunks from {len(documents)} file(s).")

    # Embed documents in batches to respect API rate limits
    print("Embedding documents in batches...")
    batch_size = 100
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(chunks) - 1)//batch_size + 1}...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        else:
            vector_store.add_documents(batch)

        # A small delay can help with very strict rate limits
        time.sleep(1)

    print("Finished embedding all documents. Database has been built and saved.")
    return vector_store.as_retriever()


# --- 3. PHASE 2: STATEFUL AGENT WITH LANGGRAPH ---

class UserPreferences(TypedDict):
    causes: List[str]
    locations: List[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str


class MockEmbeddings:
    def embed_query(self, query: str):
        # Return a fixed-length dummy vector for the query
        return [0.0] * 512

    def embed_documents(self, documents: List[Document]):
        # Return a list of dummy vectors for each document
        return [[0.0] * 512 for _ in documents]


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
retriever = None


def classify_intent_node(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
        """Given the user's latest message, classify the intent: 'preference_update', 'recommendation_request', 'question', 'greeting', 'goodbye'. Return only the single-word classification.

User Message: {user_message}"""
    )
    user_message = state["messages"][-1].content
    chain = prompt | llm
    intent = chain.invoke({"user_message": user_message}).content.strip()
    return {"latest_intent": intent}


def update_preferences_node(state: AgentState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting user preferences for philanthropic causes and locations from a message. Return a JSON object with two keys: 'causes' and 'locations', listing any extracted terms. If none, return empty lists."),
        ("human", "{user_message}")
    ])
    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser

    user_message = state["messages"][-1].content
    extracted_prefs = chain.invoke({"user_message": user_message})

    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"] = list(
        set(current_prefs["causes"] + extracted_prefs.get("causes", [])))
    current_prefs["locations"] = list(
        set(current_prefs["locations"] + extracted_prefs.get("locations", [])))

    confirmation_message = "Thanks! I've updated your preferences."
    return {"preferences": current_prefs, "messages": [HumanMessage(content=confirmation_message, name="System")]}


def retrieve_documents_node(state: AgentState):
    user_message = state["messages"][-1].content
    prefs = state.get("preferences", {})
    query = f"{user_message}"
    if prefs.get("causes"):
        query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"):
        query += f" in locations like {', '.join(prefs['locations'])}"
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

# --- KEY CHANGE: Safely handle responses with or without retrieved docs ---


def generate_response_node(state: AgentState):
    question = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs")
    preferences = state.get("preferences", {"causes": [], "locations": []})

    prefs_text = ""
    if preferences["causes"]:
        prefs_text += f"Causes of interest: {', '.join(preferences['causes'])}.\n"
    if preferences["locations"]:
        prefs_text += f"Preferred locations: {', '.join(preferences['locations'])}.\n"

    if not retrieved_docs:
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroAI, a friendly assistant. Here are the user's preferences:
{prefs}

Answer the user's question in a conversational way.

User Message: {question}
""")
        chain = prompt | llm
        response = chain.invoke({"prefs": prefs_text, "question": question})
    else:
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroAI, an AI assistant for discovering NGOs. Here is context from documents and user preferences:
{context}

{prefs}

Answer the user's question. If you don’t have enough information, say so.

User Question: {question}
""")
        chain = prompt | llm
        response = chain.invoke(
            {"context": context, "prefs": prefs_text, "question": question})
        # response = chain.invoke(
        # {"history": history, "prefs": prefs_text, "question": question})

    return {"messages": [response]}

    """
    Generates a response, adapting whether documents were retrieved or not.
    """
    question = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs")  # Safely get docs

    if not retrieved_docs:
        # If no documents, this is a general chat interaction (like a greeting)
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroAI, a helpful and friendly AI assistant. Provide a simple, conversational response to the user's message.

User Message: {question}"""
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
    else:
        # If documents were retrieved, perform RAG
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroAI, an AI assistant for discovering trustworthy NGOs. Answer the user's question based ONLY on the provided context. Be conversational and helpful. If the context doesn't contain the answer, say you don't have enough information.

**Context:**
{context}

**User Question:**
{question}"""
        )
        chain = prompt | llm
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        response = chain.invoke({"context": context, "question": question})

    return {"messages": [response]}


def route_after_classification(state: AgentState):
    intent = state["latest_intent"]
    if intent == "goodbye":
        return END
    if intent == "preference_update":
        return "update_preferences"
    if intent in ["question", "recommendation_request"]:
        return "retrieve_documents"
    return "generate_response"  # For greetings etc.


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("update_preferences", update_preferences_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {"update_preferences": "update_preferences", "retrieve_documents": "retrieve_documents",
            "generate_response": "generate_response", END: END}
    )
    workflow.add_edge("update_preferences", END)
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile()


# --- Main Interaction Loop ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhilanthroAI RAG Assistant")
    parser.add_argument('--rebuild-db', action='store_true',
                        help="Force deletion and rebuilding of the vector database.")
    args = parser.parse_args()

    if args.rebuild_db and os.path.exists(DB_DIR):
        print(
            f"Rebuild flag detected. Deleting existing database at '{DB_DIR}'...")
        shutil.rmtree(DB_DIR)

    print("Setting up PhilanthroAI...")
    retriever = build_rag_pipeline()

    if retriever:
        app = build_graph()
        print("\nPhilanthroAI is ready! How can I help you find an NGO to support?")
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})

        state = {
            "messages": [],
            "preferences": {"causes": [], "locations": []},
            "retrieved_docs": [],
            "latest_intent": ""
        }

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("PhilanthroAI: Goodbye!")
                break

            state["messages"].append(HumanMessage(content=user_input))

            events = app.stream(
                {"messages": state["messages"], "preferences": state["preferences"]}, config=config)

            final_message = None
            for event in events:
                updated_state = None  # Initialize at start of each loop iteration

                if "generate_response" in event:
                    updated_state = event["generate_response"]
                    final_message = updated_state["messages"][-1]
                elif "update_preferences" in event:
                    updated_state = event["update_preferences"]
                    final_message = updated_state["messages"][-1]

                # Only update state if updated_state is not None
                if updated_state:
                    state["preferences"] = updated_state.get(
                        "preferences", state["preferences"])
                    state["messages"].extend(updated_state.get("messages", []))

            if final_message:
                print(f"PhilanthroAI: {final_message.content}")
