"""
Consolidated Examples for Local RAG System
Contains all example code in one organized file
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def example_chunking():
    """Example of text chunking with LangChain"""
    print("=== Text Chunking Example ===")
    
    # Sample text from LangChain documentation
    entire_text = """
LangChain is a framework for developing applications powered by large language models (LLMs).

LangChain simplifies every stage of the LLM application lifecycle:

Development: Build your applications using LangChain's open-source components and third-party integrations. Use LangGraph to build stateful agents with first-class streaming and human-in-the-loop support.
Productionization: Use LangSmith to inspect, monitor and evaluate your applications, so that you can continuously optimize and deploy with confidence.
Deployment: Turn your LangGraph applications into production-ready APIs and Assistants with LangGraph Platform.

LangChain implements a standard interface for large language models and related technologies, such as embedding models and vector stores, and integrates with hundreds of providers.

Architecture
The LangChain framework consists of multiple open-source libraries:
langchain-core: Base abstractions for chat models and other components.
Integration packages (e.g. langchain-openai, langchain-anthropic, etc.): Important integrations have been split into lightweight packages.
langchain: Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
langchain-community: Third-party integrations that are community maintained.
langgraph: Orchestration framework for combining LangChain components into production-ready applications.
    """
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Create documents
    texts = text_splitter.create_documents([entire_text])
    
    print(f"Total chunks: {len(texts)}")
    print(f"First chunk: {texts[0].page_content}")
    
    return texts

def example_embedding():
    """Example of creating embeddings with Ollama"""
    print("\n=== Embedding Example ===")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    
    # Sample text
    text_to_embed = "LangChain is a framework for developing applications powered by large language models (LLMs)."
    
    # Create embedding
    single_vector = embeddings.embed_query(text_to_embed)
    
    print(f"Text: {text_to_embed}")
    print(f"Embedding dimension: {len(single_vector)}")
    print(f"First 5 values: {single_vector[:5]}")
    
    return single_vector

def example_retriever():
    """Example of document retrieval with similarity search"""
    print("\n=== Document Retrieval Example ===")
    
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    
    # Sample documents for demonstration
    sample_docs = [
        "LangChain is a framework for developing applications powered by large language models.",
        "LangChain simplifies every stage of the LLM application lifecycle.",
        "LangGraph is used to build stateful agents with streaming support.",
        "LangSmith helps inspect, monitor and evaluate applications.",
        "LangChain integrates with hundreds of providers and technologies."
    ]
    
    # Create embeddings for sample documents
    doc_embeddings = embeddings.embed_documents(sample_docs)
    
    # Search for similar documents
    query = "what is langchain"
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities
    import numpy as np
    similarities = []
    for i, doc_embedding in enumerate(doc_embeddings):
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        similarities.append((similarity, sample_docs[i]))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    print(f"Query: {query}")
    print(f"Found {len(similarities)} results:")
    
    for i, (similarity, doc) in enumerate(similarities):
        print(f"{i+1}. [Similarity: {similarity:.3f}] {doc}")
    
    return similarities

def example_complete_pipeline():
    """Example of complete RAG pipeline"""
    print("\n=== Complete RAG Pipeline Example ===")
    
    # Step 1: Text chunking
    print("Step 1: Text Chunking")
    chunks = example_chunking()
    
    # Step 2: Embeddings
    print("\nStep 2: Creating Embeddings")
    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))
    
    # Create embeddings for chunks
    chunk_texts = [chunk.page_content for chunk in chunks[:3]]  # Use first 3 chunks
    chunk_embeddings = embeddings.embed_documents(chunk_texts)
    
    print(f"Created embeddings for {len(chunk_texts)} chunks")
    
    # Step 3: Similarity search
    print("\nStep 3: Similarity Search")
    query = "LangChain framework"
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities
    import numpy as np
    similarities = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        similarity = np.dot(query_embedding, chunk_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append((similarity, i, chunk_texts[i]))
    
    # Sort by similarity
    similarities.sort(reverse=True, key=lambda x: x[0])
    
    print(f"Query: {query}")
    print("Most similar chunks:")
    for similarity, idx, text in similarities[:2]:
        print(f"  Similarity: {similarity:.3f}")
        print(f"  Text: {text[:100]}...")
    
    return similarities

def main():
    """Run all examples"""
    print("🚀 Running Local RAG Examples")
    print("=" * 50)
    
    try:
        # Run individual examples
        example_chunking()
        example_embedding()
        
        # Try retrieval example (might fail if no data exists)
        try:
            example_retriever()
        except Exception as e:
            print(f"Retrieval example failed (expected if no data exists): {e}")
        
        # Run complete pipeline example
        example_complete_pipeline()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
