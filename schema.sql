CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    processed BOOLEAN DEFAULT FALSE,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL
);

CREATE TABLE edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES nodes(id),
    target_id INTEGER REFERENCES nodes(id),
    relationship TEXT NOT NULL,
    weight FLOAT NOT NULL,
    UNIQUE (source_id, target_id, relationship)
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    document_id INTEGER REFERENCES documents(id),
    embedding VECTOR(384)
);

CREATE TABLE chunk_entities (
    chunk_id INTEGER REFERENCES chunks(id),
    entity_id INTEGER REFERENCES nodes(id),
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE TABLE communities (
    id INTEGER PRIMARY KEY,
    nodes TEXT[] NOT NULL,
    summary TEXT NOT NULL,
    summary_embedding VECTOR(384)
);