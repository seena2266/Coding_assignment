# retrieval.py
# This script handles the retrieval step of a RAG (Retrieval-Augmented Generation)
# pipeline. It loads a pre-built FAISS index, a metadata mapping, and an
# embedding model. It then performs a similarity search on the index based on
# a user's query, retrieves the corresponding data, and handles image processing.

# Imports
# Standard library imports for data handling and file operations.
import pickle
import faiss
import numpy as np
import os
# Third-party imports.
# sentence_transformers: For creating a vector representation of the query.
from sentence_transformers import SentenceTransformer
# PIL: For handling image resizing.
from PIL import Image

# -------- Paths and Config --------
# Paths to the pre-built index and metadata files.
INDEX_PATH = "data/myindex.faiss"
MAPPING_PATH = "data/id_mapping.pkl"
# The name of the SentenceTransformer model used for embedding.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ----------------------------------


def load_resources():

    """
    Functionality:
        Loads the necessary resources for the retrieval process:
        - The FAISS index for vector search.
        - The metadata mapping (a Python dictionary) that links vector IDs
          back to the original data.
        - The SentenceTransformer model for embedding the query.

    Returns:
        tuple: A tuple containing the loaded index, ID mapping, and model.
               Returns `(None, None, None)` if any resource fails to load.
    """

    try:
        index = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, "rb") as f:
            id_mapping = pickle.load(f)
        model = SentenceTransformer(MODEL_NAME)
        return index, id_mapping, model
    except (FileNotFoundError, faiss.FaissError, pickle.PickleError) as e:
        # For silent operation, use a logging library or pass to a UI element.
        return None, None, None


def resize_image(image_path, size=(225, 225)):

    """
    Resize an image to a specified size.

    Args:
        image_path (str): The path to the image file.
        size (tuple): A tuple (width, height) for the new image size.

    Returns:
        PIL.Image: The resized image object, or None if an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize(size)
            return img
    except (FileNotFoundError, OSError) as e:
        return None


def search(query, top_k=1):

    """
    Functionality:
        Performs the main search operation. It takes a user query, embeds it,
        and finds the most similar vectors in the FAISS index. It then uses
        the metadata mapping to retrieve the full original data and the
        corresponding image.

    Args:
        query (str): The user's search query.
        top_k (int): The number of top results to retrieve.

    Returns:
        list or None: A list of dictionaries, where each dictionary contains
                      the retrieved data (question, answer, image) for a
                      matching entry. Returns None if resources could not be loaded.
    """
    
    index, id_mapping, model = load_resources()
    if index is None or id_mapping is None or model is None:
        return None

    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Search FAISS
    D, I = index.search(query_emb, top_k)

    results = []
    for idx in I[0]:
        if idx == -1:  # Handle cases where no match is found
            continue
        
        entry = id_mapping[idx]

        question = entry.get("question", "")
        answer = entry.get("answer", "")
        image_path = entry.get("_linked_sample_data", {}).get("filename", "")
        
        full_image_path = os.path.join("data/nuscenes", image_path)

        resized_img = resize_image(full_image_path)
        
        results.append(
            {
                "question": question,
                "answer": answer,
                "image_path": image_path,
                "resized_image": resized_img,
                "attributes": entry.get("_annotations", []),
                "linked_sample": entry.get("_linked_sample", {}),
                "ego_pose": entry.get("_ego_pose", {}),
            }
        )
    return results[0] if results else None