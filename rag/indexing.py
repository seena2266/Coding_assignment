# indexing.py
# This script is responsible for building a FAISS index from a dataset for a
# Retrieval-Augmented Generation (RAG) system. It loads question-answer pairs,
# converts them into vector embeddings, and saves both the FAISS index and
# a metadata mapping file.

# Imports
# Standard library imports.
import json
import os
import pickle

# Third-party imports for vector indexing, numerical operations, and NLP.
# faiss: A library for efficient similarity search and clustering.
import faiss
# numpy: A fundamental package for scientific computing with Python.
import numpy as np
# sentence_transformers: A library for generating sentence embeddings.
from sentence_transformers import SentenceTransformer
# tqdm: A library to display a smart progress bar for loops.
from tqdm import tqdm


def load_json(input_file):

    """Load parsed_nuscenes_driveLM.json file."""

    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(data, model_name, batch_size=64):

    """
    Build embeddings for Q&A pairs only.
    Full entries are stored separately in metadata mapping.
    """

    model = SentenceTransformer(model_name)

    qa_texts = []
    for item in data:
        q = item.get("question", "")
        a = item.get("answer", "")
        qa_texts.append(f"Q: {q} A: {a}")

    embeddings = model.encode(
        qa_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # important for cosine similarity
    )
    return embeddings


def save_faiss_index(embeddings, index_out):

    """Save FAISS index (cosine similarity)."""

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (since embeddings are normalized)
    index.add(embeddings)
    faiss.write_index(index, index_out)
    print(f"FAISS index saved to {index_out}")


def save_metadata(data, meta_out):
    
    """
    Save full metadata (not just Q&A) in pickle format.
    Each FAISS ID maps to one full entry from parsed JSON.
    """
    id_mapping = {idx: entry for idx, entry in enumerate(data)}
    with open(meta_out, "wb") as f:
        pickle.dump(id_mapping, f)
    print(f"Metadata mapping saved to {meta_out}")


def main():
    
    """
    Functionality:
        The main execution function to orchestrate the entire indexing process.
        It defines the input and output paths, loads the data, builds the
        embeddings, and saves both the FAISS index and the associated metadata.
    """
    # -------- Paths -------- #
    input_file = "data/drivelm/parsed_nuscenes_driveLM.json"
    index_out = "data/myindex.faiss"
    meta_out = "data/id_mapping.pkl"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size = 64
    # ----------------------- #

    # Ensure output folder exists
    os.makedirs(os.path.dirname(index_out), exist_ok=True)

    print("Loading parsed JSON...")
    data = load_json(input_file)

    print(f"Extracted {len(data)} entries")

    print("Building embeddings...")
    embeddings = build_embeddings(data, model_name, batch_size)

    print(" Saving FAISS index...")
    save_faiss_index(np.array(embeddings), index_out)

    print(" Saving ID mapping...")
    save_metadata(data, meta_out)

    print(" Indexing complete!")


if __name__ == "__main__":
    main()