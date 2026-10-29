"""Pipeline RAG : question -> recherche PGVector -> Qwen (Ollama) -> réponse sourcée.
Usage : python rag.py "Qu'est-ce que le multi-head attention ?"
"""
import sys

import ollama
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DSN = "host=localhost port=5432 dbname=ragdb user=rag password=rag"
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CLIP_MODEL = "clip-ViT-B-32"
LLM = "qwen2.5:3b"
TOP_K = 5

SYSTEM_PROMPT = (
    "Tu es un assistant qui répond à des questions sur un document PDF. "
    "Réponds UNIQUEMENT à partir du CONTEXTE fourni. "
    "Si la réponse n'est pas dans le contexte, dis clairement que tu ne la trouves pas. "
    "Cite les pages utilisées sous la forme [p.4]. "
    "Réponds dans la langue de la question, de façon concise."
)

# Modèles et connexion chargés une seule fois
_text_model = SentenceTransformer(TEXT_MODEL)
_clip = SentenceTransformer(CLIP_MODEL)
_conn = psycopg2.connect(DSN)
register_vector(_conn)


def retrieve(question, k=TOP_K):
    """Retrouve les k passages/tableaux les plus proches + 2 images (CLIP)."""
    cur = _conn.cursor()
    cur.execute(
        "SELECT type, page, content FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (_text_model.encode(question), k))
    chunks = cur.fetchall()
    cur.execute(
        "SELECT page, image_path FROM images ORDER BY clip_embedding <=> %s LIMIT 2",
        (_clip.encode(question),))
    images = cur.fetchall()
    return chunks, images


def answer(question):
    chunks, images = retrieve(question)
    # Correction de la f-string sur une seule ligne pour éviter l'erreur de syntaxe
    context = "\n\n".join(f"[Page {p} - {t}]\n{c}" for t, p, c in chunks)
    resp = ollama.chat(
        model=LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"CONTEXTE :\n{context}\n\nQUESTION : {question}"},
        ],
        options={"temperature": 0.1},
    )
    pages = sorted({p for _, p, _ in chunks})
    return resp["message"]["content"].strip(), pages, images


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is multi-head attention?"
    text, pages, images = answer(q)
    print("\n=== RÉPONSE ===\n" + text)
    print("\nPages consultées :", pages)
    print("Images pertinentes :", [path for _, path in images])