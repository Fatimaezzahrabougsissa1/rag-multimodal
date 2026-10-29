"""Indexation multimodale d'un PDF dans PGVector.

PDF -> LlamaIndex (texte + chunking) / PyMuPDF (images) / pdfplumber (tableaux)
    -> embeddings texte (MiniLM multilingue) + CLIP (images)
    -> PostgreSQL + pgvector
Usage : python ingestion.py mon_document.pdf [--describe]
"""
import io
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import ollama
import pdfplumber
import psycopg2
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PyMuPDFReader
from PIL import Image
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

DSN = "host=localhost port=5432 dbname=ragdb user=rag password=rag"
TEXT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 dims, léger, FR/EN
CLIP_MODEL = "clip-ViT-B-32"                            # 512 dims, texte + images
VLM_MODEL = "qwen2.5vl:3b"                              # via Ollama (descriptions)
IMG_DIR = Path("data/images")
MIN_SIZE = 100  # ignorer les icônes / petits logos (pixels)

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    doc TEXT, page INT,
    type TEXT,               -- 'text' | 'table' | 'image_description'
    content TEXT,
    image_path TEXT,         -- renseigné pour image_description
    embedding vector(384)
);
CREATE TABLE IF NOT EXISTS images (
    id SERIAL PRIMARY KEY,
    doc TEXT, page INT,
    image_path TEXT,
    description TEXT,
    clip_embedding vector(512)
);
"""


def table_to_markdown(table):
    rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in table]
    if not rows:
        return ""
    lines = ["| " + " | ".join(rows[0]) + " |",
             "|" + "---|" * len(rows[0])]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def describe_image(path):
    resp = ollama.chat(
        model=VLM_MODEL,
        messages=[{
            "role": "user",
            "content": "Décris précisément cette figure ou ce schéma technique "
                       "en 2-3 phrases (éléments, légendes, relations).",
            "images": [str(path)],
        }],
    )
    return resp["message"]["content"].strip()


def main(pdf_path, describe=False):
    pdf_path = Path(pdf_path)
    doc_name = pdf_path.name
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(DSN)
    cur = conn.cursor()
    cur.execute(SCHEMA)
    conn.commit()
    register_vector(conn)

    text_model = SentenceTransformer(TEXT_MODEL)
    clip = SentenceTransformer(CLIP_MODEL)

    # 1) TEXTE : LlamaIndex (lecture + découpage en passages)
    pages = PyMuPDFReader().load_data(file_path=pdf_path)
    docs = [Document(text=p.text, metadata={"page": i + 1})
            for i, p in enumerate(pages)]
    nodes = SentenceSplitter(chunk_size=512, chunk_overlap=64) \
        .get_nodes_from_documents(docs)
    texts = [n.get_content() for n in nodes]
    embs = text_model.encode(texts, show_progress_bar=True)
    for n, t, e in zip(nodes, texts, embs):
        cur.execute(
            "INSERT INTO chunks (doc,page,type,content,embedding) VALUES (%s,%s,'text',%s,%s)",
            (doc_name, n.metadata["page"], t, e))
    print(f"{len(nodes)} passages texte indexés")

    # 2) TABLEAUX : pdfplumber -> markdown
    n_tab = 0
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            for tab in page.extract_tables():
                md = table_to_markdown(tab)
                if len(md) < 20:
                    continue
                cur.execute(
                    "INSERT INTO chunks (doc,page,type,content,embedding) VALUES (%s,%s,'table',%s,%s)",
                    (doc_name, i + 1, md, text_model.encode(md)))
                n_tab += 1
    print(f"{n_tab} tableaux indexés")

    # 3) IMAGES : PyMuPDF -> CLIP (+ description VLM optionnelle)
    n_img = 0
    pdf = fitz.open(pdf_path)
    for pno in range(len(pdf)):
        for k, info in enumerate(pdf[pno].get_images(full=True)):
            raw = pdf.extract_image(info[0])
            img = Image.open(io.BytesIO(raw["image"])).convert("RGB")
            if min(img.size) < MIN_SIZE:
                continue
            path = IMG_DIR / f"{pdf_path.stem}_p{pno + 1}_{k}.png"
            img.save(path)
            clip_emb = clip.encode(img)
            desc = describe_image(path) if describe else ""
            cur.execute(
                "INSERT INTO images (doc,page,image_path,description,clip_embedding) VALUES (%s,%s,%s,%s,%s)",
                (doc_name, pno + 1, str(path), desc, clip_emb))
            if desc:  # description rendue interrogeable via le texte
                cur.execute(
                    "INSERT INTO chunks (doc,page,type,content,image_path,embedding) "
                    "VALUES (%s,%s,'image_description',%s,%s,%s)",
                    (doc_name, pno + 1, desc, str(path), text_model.encode(desc)))
            n_img += 1
    print(f"{n_img} images indexées")

    conn.commit()
    conn.close()
    print("Indexation terminée.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage : python ingestion.py fichier.pdf [--describe]")
    main(sys.argv[1], describe="--describe" in sys.argv)
