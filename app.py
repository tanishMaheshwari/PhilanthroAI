import os
import shutil
import uuid
from dotenv import load_dotenv
from typing import List, Annotated
import time
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
# --- MODIFIED IMPORT ---
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()  # Load environment variables from .env file
# Ensure you have set the GOOGLE_API_KEY environment variable
if 'GOOGLE_API_KEY' not in os.environ:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

# Define directories
PROFILE_DIR = "./docs"
DB_DIR = "./chroma_db"

# --- 2. PHASE 1: KNOWLEDGE BASE CONSTRUCTION (FOR PDFs) ---

# ... (keep the rest of your imports)

def build_rag_pipeline():
    """
    Builds the RAG pipeline by loading PDF files, splitting, and indexing them.
    Returns a retriever object.
    """
    if not os.path.exists(PROFILE_DIR):
        print(f"Error: Profile directory '{PROFILE_DIR}' not found. Please create it and add your PDF files.")
        return None

    loader = DirectoryLoader(
        PROFILE_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        print("\n--- ERROR ---")
        print(f"No PDF documents were found in the '{PROFILE_DIR}' directory.")
        print("Please ensure your PDF files are in the correct folder.")
        print("---------------")
        exit()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} document chunks.") # Good to know how many we're dealing with

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # --- START OF MODIFIED SECTION ---
    print("Embedding documents in batches to respect API rate limits...")
    batch_size = 100 # A reasonable batch size for the free tier
    vector_store = None # Initialize the vector store

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(chunks) - 1)//batch_size + 1}...")

        if vector_store is None:
            # Create the store with the first batch
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        else:
            # Add subsequent batches to the existing store
            vector_store.add_documents(batch)
        
        # Add a delay between batches to avoid hitting the per-minute limit
        time.sleep(1) # Wait for 5 seconds between batches

    print("Finished embedding all documents.")
    # --- END OF MODIFIED SECTION ---

    print(f"RAG pipeline built: {len(documents)} PDF document(s) loaded and indexed.")
    return vector_store.as_retriever()

# --- 3. PHASE 2: STATEFUL AGENT WITH LANGGRAPH ---

class UserPreferences(TypedDict):
    """Structure to hold the user's learned preferences."""
    causes: List[str]
    locations: List[str]

class AgentState(TypedDict):
    """Defines the main state for the entire graph."""
    messages: Annotated[list, add_messages]
    preferences: UserPreferences
    retrieved_docs: List[Document]
    latest_intent: str

# Define the LLM and build the retriever
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5) # Using gemini-2.0-flash as it's a newer model
retriever = None # Will be initialized in the main block

# --- Graph Nodes (No changes needed here) ---

def classify_intent_node(state: AgentState):
    """Analyzes the latest user message to determine its purpose."""
    prompt = ChatPromptTemplate.from_template(
        """Given the user's latest message, classify the intent:
'preference_update', 'recommendation_request', 'question', 'greeting', 'goodbye'.
Return only the single-word classification.

User Message: {user_message}"""
    )
    user_message = state["messages"][-1].content
    chain = prompt | llm
    intent = chain.invoke({"user_message": user_message}).content.strip()
    return {"latest_intent": intent}

def update_preferences_node(state: AgentState):
    """Parses the user's message to extract and store preferences."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting user preferences for philanthropic causes and locations from a message.
Return a JSON object with two keys: 'causes' and 'locations', listing any extracted terms. If none, return empty lists."""),
        ("human", "{user_message}")
    ])
    parser = JsonOutputParser(pydantic_object=UserPreferences)
    chain = prompt | llm | parser

    user_message = state["messages"][-1].content
    extracted_prefs = chain.invoke({"user_message": user_message})

    current_prefs = state.get("preferences", {"causes": [], "locations": []})
    current_prefs["causes"] = list(set(current_prefs["causes"] + extracted_prefs.get("causes", [])))
    current_prefs["locations"] = list(set(current_prefs["locations"] + extracted_prefs.get("locations", [])))

    confirmation_message = "Thanks! I've updated your preferences."
    return {
        "preferences": current_prefs,
        "messages": [HumanMessage(content=confirmation_message, name="System")]
    }

def retrieve_documents_node(state: AgentState):
    """Constructs a query and retrieves relevant documents from the vector store."""
    user_message = state["messages"][-1].content
    prefs = state.get("preferences", {})
    query = f"{user_message}"
    if prefs.get("causes"): query += f" related to causes like {', '.join(prefs['causes'])}"
    if prefs.get("locations"): query += f" in locations like {', '.join(prefs['locations'])}"
    docs = retriever.invoke(query)
    return {"retrieved_docs": docs}

def generate_response_node(state: AgentState):
    """
    Generates a response, adapting whether documents were retrieved or not.
    """
    question = state["messages"][-1].content
    # Use .get() to safely access retrieved_docs, it returns None if the key doesn't exist
    retrieved_docs = state.get("retrieved_docs")

    if not retrieved_docs:
        # If no documents are present, this is a general chat interaction
        prompt = ChatPromptTemplate.from_template(
            """You are PhilanthroBot, a helpful and friendly AI assistant.
            Provide a simple, conversational response to the user's message.

            User Message: {question}"""
        )
        chain = prompt | llm
        response = chain.invoke({"question": question})
    else:
        # If documents were retrieved, perform RAG to answer the question
        prompt = ChatPromptTemplate.from_template(
"""You are PhilanthroBot, a helpful AI assistant for discovering trustworthy NGOs.
Answer the user's question based ONLY on the provided context. Be conversational and helpful.
If the context doesn't contain the answer, state that you don't have enough information.

**Context:**
{context}

**User Question:**
{question}"""
        )
        chain = prompt | llm
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        response = chain.invoke({"context": context, "question": question})

    return {"messages": [response]}
# --- Conditional Edges (No changes needed here) ---

def route_after_classification(state: AgentState):
    """Decides the next step based on the classified intent."""
    intent = state["latest_intent"]
    if intent == "goodbye": return END
    if intent == "preference_update": return "update_preferences"
    if intent in ["question", "recommendation_request"]: return "retrieve_documents"
    return "generate_response"

# --- Build the Graph ---

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
        {"update_preferences": "update_preferences", "retrieve_documents": "retrieve_documents", "generate_response": "generate_response", END: END}
    )
    workflow.add_edge("update_preferences", END)
    workflow.add_edge("retrieve_documents", "generate_response")
    workflow.add_edge("generate_response", END)
    return workflow.compile()


# --- Main Interaction Loop ---

if __name__ == "__main__":
    print("Setting up PhilanthroBot...")
    retriever = build_rag_pipeline()
    
    if retriever:
        app = build_graph()
        print("\nPhilanthroBot is ready! How can I help you find an NGO to support?")
        thread_id = str(uuid.uuid4())
        config = RunnableConfig(configurable={"thread_id": thread_id})
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("PhilanthroBot: Goodbye!")
                break
            events = app.stream({"messages": [HumanMessage(content=user_input)]}, config=config)
            final_message = None
            for event in events:
                if "generate_response" in event:
                    final_message = event["generate_response"]["messages"][-1]
                elif "update_preferences" in event:
                    final_message = event["update_preferences"]["messages"][-1]
            if final_message:
                print(f"PhilanthroBot: {final_message.content}")
