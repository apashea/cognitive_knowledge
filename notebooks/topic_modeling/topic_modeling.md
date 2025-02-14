# Graph RAG with Obsidian Knowledge Base
## A Code-Flow Technical Documentation

## 1. Overview
This system converts an Obsidian knowledge base into a queryable Graph RAG system using local LLM capabilities. It combines graph structure awareness with semantic search to provide context-rich responses to queries about your knowledge base.

## 2. System Setup

### 2.1 Dependencies and Imports
```python
import obsidiantools.api as otools
import networkx as nx
import numpy as np
from ollama import Client
from typing import Dict, List
import os
```

Key dependencies:
- obsidiantools: Interfaces with Obsidian vault structure
- networkx: Handles graph operations and analysis
- numpy: Manages vector operations for embeddings
- ollama: Provides local LLM functionality
- typing: Enables type hints for better code clarity
- os: Handles file path operations

### 2.2 Client Initialization
```python
ollama_client = Client()
```

Initializes the Ollama client for local LLM operations. This client will be used for:
- Generating embeddings
- Processing queries
- Returning responses

## 3. Core Functions

### 3.1 Graph Loading
```python
def load_obsidian_graph(vault_path: str) -> tuple[nx.Graph, Dict[str, str]]:
    vault = otools.Vault(vault_path).connect()
    G = vault.graph
    nodes_content = {}
    for node in G.nodes():
        file_path = os.path.join(vault_path, f"{node}.md")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                nodes_content[node] = f.read()
    return G, nodes_content
```

This function:
- Connects to the Obsidian vault
- Extracts the graph structure
- Loads content for each node
- Returns both graph structure and content dictionary

### 3.2 Embedding Generation
```python
def create_embeddings(text: str) -> np.ndarray:
    response = ollama_client.embeddings(
        model='llama2',
        prompt=text
    )
    return np.array(response['embedding'])
```

Creates vector embeddings using:
- Local Llama 2 model
- Ollama's embedding endpoint
- Returns numpy array for vector operations

### 3.3 Graph RAG Construction
```python
def build_graph_rag(G: nx.Graph, nodes_content: Dict[str, str]) -> Dict[str, dict]:
    graph_rag = {}
    for node, content in nodes_content.items():
        neighbors = list(G.neighbors(node))
        graph_rag[node] = {
            'content': content,
            'embedding': create_embeddings(content),
            'neighbors': neighbors
        }
    return graph_rag
```

Builds the knowledge graph by:
- Processing each node
- Storing content and embeddings
- Maintaining neighbor relationships

### 3.4 Similarity Search
```python
def similarity_search(query: str, graph_rag: Dict[str, dict], k: int = 3) -> List[str]:
    query_embedding = create_embeddings(query)
    similarities = {}
    for node, data in graph_rag.items():
        similarity = np.dot(query_embedding, data['embedding'])
        similarities[node] = similarity
    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:k]
```

Implements semantic search through:
- Query embedding generation
- Cosine similarity calculation
- Top-k similar nodes retrieval

### 3.5 Context Retrieval
```python
def retrieve_context(query: str, graph_rag: Dict[str, dict], k: int = 3) -> str:
    similar_nodes = similarity_search(query, graph_rag, k)
    context = []
    for node, score in similar_nodes:
        context.append(f"Content: {graph_rag[node]['content']}")
        for neighbor in graph_rag[node]['neighbors']:
            if neighbor in graph_rag:
                context.append(f"Related content: {graph_rag[neighbor]['content']}")
    return "\n\n".join(context)
```

Gathers relevant context by:
- Finding similar nodes
- Including neighbor content
- Formatting context for LLM input

### 3.6 Query Processing
```python
def query_graph_rag(query: str, graph_rag: Dict[str, dict]) -> str:
    context = retrieve_context(query, graph_rag)
    prompt = f"""Context from knowledge graph:
    {context}
    
    Question: {query}
    
    Answer based on the context provided:"""
    response = ollama_client.generate(
        model='llama2',
        prompt=prompt,
        stream=False
    )
    return response['response']
```

Handles query processing through:
- Context retrieval
- Prompt construction
- LLM response generation

## 4. System Usage

### 4.1 Main Implementation
```python
def main():
    vault_path = "path/to/your/vault"
    G, nodes_content = load_obsidian_graph(vault_path)
    graph_rag = build_graph_rag(G, nodes_content)
    query = "What are the main concepts in my knowledge graph?"
    answer = query_graph_rag(query, graph_rag)
    print(answer)
```

System initialization and usage:
1. Set vault path
2. Load graph and content
3. Build Graph RAG structure
4. Process queries
5. Display results

## 5. Best Practices & Considerations

### 5.1 Performance Optimization
- Cache embeddings for frequently accessed nodes
- Implement batch processing for large graphs
- Consider pruning irrelevant nodes

### 5.2 Memory Management
- Monitor embedding storage
- Implement lazy loading for large vaults
- Clear unused embeddings

### 5.3 Query Optimization
- Structure queries clearly
- Use specific questions
- Consider context window limitations

### 5.4 System Maintenance
- Regular embedding updates
- Graph structure validation
- Content synchronization checks

### 5.5 Error Handling
- Validate file paths
- Check for missing content
- Monitor LLM responses

## 6. Future Enhancements
- Embedding persistence
- Advanced graph traversal
- Custom similarity metrics
- Parallel processing
- Alternative LLM support