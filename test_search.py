"""Test rapide de la recherche vectorielle (avant de construire l'agent)."""
import sys

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

from ingestion import DSN, TEXT_MODEL, CLIP_MODEL

question = " ".join(sys.argv[1:]) or "What is multi-head attention?"
conn = psycopg2.connect(DSN)
register_vector(conn)
cur = conn.cursor()

q = SentenceTransformer(TEXT_MODEL).encode(question)
cur.execute("SELECT type, page, LEFT(content, 150) FROM chunks "
            "ORDER BY embedding <=> %s LIMIT 5", (q,))
print("--- Passages / tableaux ---")
for row in cur.fetchall():
    print(row)

qc = SentenceTransformer(CLIP_MODEL).encode(question)
cur.execute("SELECT page, image_path FROM images "
            "ORDER BY clip_embedding <=> %s LIMIT 3", (qc,))
print("--- Images (CLIP) ---")
for row in cur.fetchall():
    print(row)
