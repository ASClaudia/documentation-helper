import os
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OllamaEmbeddings(model="nomic-embed-text", temperature=0.2, dimensions=1024)

# Initialize vector store

# a locally created folder and db - it uses sqlite3, with Chroma DB
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-index-cla", embedding=embeddings
)
# Initialize chat model
# model = init_chat_model("gpt-5.2", model_provider="openai")
model = init_chat_model("qwen3:1.7b", model_provider="ollama")

# if the response_format is content_and_artifact, the tool can return both a string (content) and an artifact
# (e.g., list of documents)
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    # this will help the llm to determine if it should use this tool or not
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents, after similarity search,
    # the vector store will return a list of Document objects with metadata and page_content
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # or we can directly call similarity_search, it will return the same list of Document objects, but it will be less
    # nicely formatted in langsmith
    # retrieved_docs = vectorstore.similarity_search(query, k=4)

    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Return both serialized content and raw documents
    return serialized, retrieved_docs


# THIS is a very simple ReAct agent.
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.

    Args:
        query: The user's question

    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # it runs langgraph under the hood
    agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

    # Build messages list
    messages = [{"role": "user", "content": query}]

    # Invoke the agent
    response = agent.invoke({"messages": messages})

    # Extract the answer from the last AI message
    answer = response["messages"][-1].content

    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    # Showing the user where the response come from, showing the links, help to generate trust. It's part of agentic
    # user experience.
    return {
        "answer": answer,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    print("STOP")