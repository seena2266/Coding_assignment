# generation.py
# This script contains the core logic for the RAG (Retrieval-Augmented Generation) system.
# It uses a Vision-Language Model (Moondream2) to process images and a Large Language
# Model (TinyLlama) to handle text-based questions, combining their outputs for a
# comprehensive answer. It also includes functions for model loading and text
# truncation to manage token limits.

# Imports
# streamlit: For building the web application interface.
import streamlit as st
# torch: PyTorch, the deep learning framework for model operations.
import torch
# transformers: Hugging Face library for loading models and tokenizers.
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# PIL: Python Imaging Library for handling images.
from PIL import Image
# json: For handling JSON data.
import json
# typing: For type hints to improve code readability and maintainability.
from typing import Tuple
# math: For mathematical operations.
import math
# gpt4all: A library for running local, open-source LLMs.
from gpt4all import GPT4All

# --- Config: tune these if necessary ---
# The identifier for the Vision-Language Model (VLM).
MODEL_ID = "vikhyatk/moondream2"
# The maximum number of tokens allowed for the text portion of the prompt.
MAX_TEXT_TOKENS = 1200   # safe cap for text portion (leave room under 2048)
# A more aggressive token cap used as a fallback for the VLM.
AGGRESSIVE_TOKENS = 600  # fallback cap (question-only)
# --------------------------------------

# --- VLM (Moondream2) related functions ---
@st.cache_resource
def load_moondream_model():

    """
    Functionality:
        Loads and caches the Moondream2 VLM model and its tokenizer. This
        function is decorated with `@st.cache_resource` to ensure the model is
        only loaded once across multiple Streamlit runs, saving resources.

    Returns:
        tuple: A tuple containing the loaded model, tokenizer, and the device
               the model is running on.
    """

    model_id = MODEL_ID
    # Choose dtype: prefer bfloat16 on supported GPUs; on CPU float16 may be okay
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    # Keep model on CPU by default for low-memory systems; if GPU available, keep it there
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, device

moon_model, moon_tokenizer, moon_model_device = load_moondream_model()

def compute_relative_objects(ego_translation, annotations):

    """
    Compute relative distance and horizontal position (left/right/straight)
    of objects w.r.t. ego-vehicle.
    """

    objects = []
    for ann in annotations[:3]:  # limit to first 3 for readability
        obj_type = ann.get("category_name", "unknown")
        obj_pos = ann.get("translation", [0, 0, 0])
        rel_x = obj_pos[0] - ego_translation[0]
        rel_y = obj_pos[1] - ego_translation[1]
        rel_distance = round(math.sqrt(rel_x ** 2 + rel_y ** 2), 2)
        # Determine horizontal relation
        if rel_x < -1:
            rel_pos = "slightly left"
        elif rel_x > 1:
            rel_pos = "slightly right"
        else:
            rel_pos = "directly ahead"

        attributes = ", ".join(ann.get("attributes", [])) or "None"

        objects.append({
            "type": obj_type,
            "pos": obj_pos,
            "attributes": attributes,
            "rel_pos": rel_pos,
            "distance": rel_distance
        })
    return objects

def _compact_context(context: dict) -> str:

    """
    Summarize ego pose, linked samples, and key objects with distance & relative position.
    """
    ego_pose = context.get("ego_pose", {})
    linked = context.get("linked_sample", {})
    annotations = context.get("attributes", [])  # corrected key

    ego_translation = ego_pose.get("translation", [0, 0, 0])
    ego_rotation = ego_pose.get("rotation", [0, 0, 0, 1])

    # Ego state summary
    ego_summary = f"Pos: {ego_translation}, Rot: {ego_rotation}"

    # Linked sample (optional)
    linked_summary = (
        f"Linked sample keys: {list(linked.keys())[:5]}" if linked else ""
    )

    # Compute relative info for first 3 objects
    rel_objects = compute_relative_objects(ego_translation, annotations)

    obj_summaries = []
    for obj in rel_objects:
        obj_summaries.append(
            f"{obj['type']} at {obj['pos']} (Attributes: {obj['attributes']}) "
            f"{obj['rel_pos']} about {obj['distance']}m away"
        )

    ann_summary = "; ".join(obj_summaries)

    # Combine all into one compact string
    if linked_summary:
        return f"Ego Pose: {ego_summary}; {ann_summary}; {linked_summary}"
    else:
        return f"Ego Pose: {ego_summary}; {ann_summary}"

def _truncate_by_tokens(text: str, max_tokens: int) -> Tuple[str, int]:

    """
    Functionality:
        Truncates a given text string to fit within a specified token limit.

    Args:
        text (str): The input text to be truncated.
        token_cap (int): The maximum number of tokens allowed.

    Returns:
        Tuple[str, int]: A tuple containing the truncated text and the
                         final count of tokens.
    """

    # tokenizer(...) handles truncation; ensure add_special_tokens=False to avoid extra tokens
    enc = moon_tokenizer(text, truncation=True, max_length=max_tokens, add_special_tokens=False, 
                         return_tensors=None)
    ids = enc["input_ids"]
    token_count = len(ids)
    truncated = moon_tokenizer.decode(ids, skip_special_tokens=True)
    return truncated, token_count

def _ensure_pil(img):

    """If img is ndarray, convert to PIL.Image. If string path, open it."""

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    try:
        # handle numpy arrays
        import numpy as np
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
    except Exception:
        pass
    # if it's a path-like string
    if isinstance(img, str):
        return Image.open(img)
    return img  # give up, let model try

# --- LLM (TinyLlama) related functions from test.py ---
@st.cache_resource
def load_llm_model():

    """
    Functionality:
        Loads a local GPT4All model, specifically the TinyLlama model.
        This function is cached to prevent re-loading on each run.

    Returns:
        GPT4All: The loaded GPT4All model instance.
    """
    print("Loading TinyLlama model...")
    generator = pipeline(
        "text-generation", 
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    print("Model loaded successfully!")
    return generator

llm_generator = load_llm_model()

def generate_answer_llm(truncated_prompt: str, query:str):

    """
    Generate an answer using a pure LLM (TinyLlama) given a truncated prompt.
    """

    formatted_prompt = f"""<|user|>
    You are an expert AI assistant for an Autonomous Driving Assistance System (ADAS). 
    Your task is to analyze sensor data from the vehicle's perspective and provide a safety-focused 
    summary.
    Analyze the following context, which contains information about the ego vehicle's position 
    and other vehicles in the scene. Infer their status and any potential risks based 
    on their relative position and attributes.

    Context:
    {truncated_prompt}

    Question:
    {query}

    Analysis and ADAS Report:
    <|assistant|>
    """
    try:
        # The prompt is already formatted and truncated from the calling function
        # Generate the text
        response = llm_generator(
            truncated_prompt, 
            max_length=500, 
            num_return_sequences=1
        )
        generated_text = response[0]['generated_text']
        return generated_text
    except Exception as e:
        return f"An error occurred: {e}"

# --- Combined RAG function ---
def generate_answer(query: str, context: dict) -> str:

    """
    Generate an answer using Moondream2 and then TinyLlama, combining both outputs.
    This function compacts + truncates the text portion to stay under the model's window.
    """

    if moon_model is None or moon_tokenizer is None:
        return "Error: VLM model/tokenizer not loaded."
   
    # Prepare image for VLM
    image = context.get("resized_image") or context.get("image") or None
    image = _ensure_pil(image)
    if image is None:
        return "Error: Image not found in retrieved context for VLM."

    # Build compacted context string for both models
    compact_ctx = _compact_context(context)

    # Compose prompt and truncate by tokens leaving headroom for answer & image
    raw_prompt = f"Context:{compact_ctx}\nQuestion:{query}"

    # Primary truncation (safe cap)
    truncated_prompt, tcount = _truncate_by_tokens(raw_prompt, MAX_TEXT_TOKENS)

    print(truncated_prompt)

    # Show token counts (helpful for debugging in Streamlit)
    try:
        st.caption(f"Prompt tokens after truncation: {tcount}")
    except Exception:
        # If Streamlit not available, ignore
        pass

    # If still longer than model window (defensive), do aggressive fallback
    if tcount >= 2000:
        # Very unlikely because we truncated, but just in case
        fallback_prompt, fcount = _truncate_by_tokens(f"Question:{query}", AGGRESSIVE_TOKENS)
        truncated_prompt = fallback_prompt
        try:
            st.warning(f"Context was too large and was removed. Prompt tokens now: {fcount}")
        except Exception:
            pass
    
    # 1. --- VLM (Moondream2) Call ---
    vlm_answer = "VLM model could not generate an answer."
    
    vlm_raw_prompt = f"""You are an expert Multi-modal AI for an Autonomous Driving Assistance 
    System (ADAS). 
    Your role is to analyze the scene in the attached image and cross-reference it with the provided 
    structured sensor data.
    Based on this combined information, provide a clear, concise report on 
    the status of vehicles and objects in the scene and any potential safety concerns.

    [Image is attached]

    Context:
    {compact_ctx}

    Question:
    {query}

    Visual and Safety Report:
    """
    try:
        with torch.no_grad():
            enc_image = moon_model.encode_image(image)
            answer_raw = moon_model.answer_question(enc_image, truncated_prompt, moon_tokenizer)
            if isinstance(answer_raw, str):
                vlm_answer = answer_raw.strip()
            elif isinstance(answer_raw, dict) and "answer" in answer_raw:
                vlm_answer = str(answer_raw["answer"]).strip()
            else:
                vlm_answer = str(answer_raw).strip()
    except IndexError as ie:
        # Fallback for VLM call
        try:
            fallback_prompt, fcount = _truncate_by_tokens(f"Question:{query}", AGGRESSIVE_TOKENS)
            st.warning("IndexError from VLM model; retrying with question-only fallback.")
            with torch.no_grad():
                enc_image = moon_model.encode_image(image)
                answer_raw = moon_model.answer_question(enc_image, fallback_prompt, moon_tokenizer)
                if isinstance(answer_raw, str):
                    vlm_answer = answer_raw.strip()
                elif isinstance(answer_raw, dict) and "answer" in answer_raw:
                    vlm_answer = str(answer_raw["answer"]).strip()
                else:
                    vlm_answer = str(answer_raw).strip()
        except Exception as e:
            vlm_answer = f"VLM model error after fallback: {e}"
    except Exception as e:
        vlm_answer = f"VLM model error: {e}"

    # 2. --- LLM (TinyLlama) Call with text only ---
    llm_answer = "LLM model could not generate an answer."
    try:
        llm_answer = generate_answer_llm(truncated_prompt, query)
    except Exception as e:
        llm_answer = f"LLM model error: {e}"

    # 3. --- Combine Answers without labels ---
    # The change is here: simply concatenate the two answers.
    combined_answer = f"{vlm_answer}\n\n{llm_answer}"
    return combined_answer

