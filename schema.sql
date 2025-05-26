CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS communities, chunk_entities, edges, nodes, chunks, documents CASCADE;

CREATE TABLE documents ( id SERIAL PRIMARY KEY, path TEXT NOT NULL, processed BOOLEAN DEFAULT FALSE );

CREATE TABLE chunks ( id SERIAL PRIMARY KEY, text TEXT NOT NULL, embedding VECTOR(1536), document_id INTEGER REFERENCES documents(id) );

CREATE TABLE nodes ( id SERIAL PRIMARY KEY, name TEXT NOT NULL UNIQUE, type TEXT );

CREATE TABLE edges ( id SERIAL PRIMARY KEY, source_id INTEGER REFERENCES nodes(id), target_id INTEGER REFERENCES nodes(id), relationship TEXT, weight FLOAT );

CREATE TABLE chunk_entities ( chunk_id INTEGER REFERENCES chunks(id), entity_id INTEGER REFERENCES nodes(id), PRIMARY KEY (chunk_id, entity_id) );

CREATE TABLE communities ( id SERIAL PRIMARY KEY, nodes INTEGER[], summary TEXT, summary_embedding VECTOR(1536) );