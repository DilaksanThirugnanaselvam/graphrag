# GraphRAG Project

A from-scratch implementation of GraphRAG based on the research paper [GraphRAG: Knowledge Graphs for AI Applications](https://arxiv.org/pdf/2404.16130). This project supports incremental indexing, vector-based querying with `pgvector`, and community detection using the Leiden algorithm.

## Features
- Incremental indexing of text documents using a `documents` table.
- Entity extraction and graph construction with `networkx`.
- Community detection with `leidenalg` and `igraph`.
- Vector similarity searches using `pgvector` (384-dimensional embeddings).
- Global and local queries powered by `sentence-transformers` and an LLM.

## Prerequisites
- Docker Desktop (for Windows: ensure WSL2 is enabled).
- Python 3.12+ (optional for local setup).
- PostgreSQL with `pgvector` (handled via Docker).

## Setup

1. **Clone the Repository**:
   ```powershell
   git clone <repository-url>
   cd graphrag_project/graphrag