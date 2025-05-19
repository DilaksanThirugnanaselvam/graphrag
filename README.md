# GraphRAG Project

A from-scratch implementation of GraphRAG based on the research paper (https://arxiv.org/pdf/2404.16130).

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure `configs/settings.yaml` with your LLM API key and endpoint.

3. Place input text in `data/input/sample.txt` (a sample is included).

## Usage

Run indexing:
```bash
python scripts/run_indexing.py
```

Run queries:
```bash
python scripts/run_query.py
```

Run tests:
```bash
pytest tests/
```

## Project Structure

- `src/`: Core implementation (text processing, entity extraction, graph building, querying).
- `scripts/`: Entry points for indexing and querying.
- `graphrag_extender\`: Incremental graph extender package with PostgreSQL/       pgvector.
- `configs/`: Configuration files.
- `data/`: Input and output directories.
- `tests/`: Unit tests.

## License

MIT
