CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE nodes (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(50)
);

CREATE TABLE edges (
    id SERIAL PRIMARY KEY,
    source_id INTEGER REFERENCES nodes(id),
    target_id INTEGER REFERENCES nodes(id),
    relationship VARCHAR(255) NOT NULL,
    weight FLOAT NOT NULL,
    UNIQUE (source_id, target_id, relationship)
);

CREATE TABLE chunks (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
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