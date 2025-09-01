# analysis.py
# This script performs a basic analysis of the parsed DriveLM and NuScenes
# dataset. It visualizes the distribution of question types, object categories,
# and ego vehicle speed using matplotlib, saving the plots as image files.

# Standard library imports.
import json
import os
from collections import Counter

# Third-party imports for plotting and data manipulation.
# matplotlib.pyplot: A state-based interface for creating plots.
import matplotlib.pyplot as plt


# Global variable for the output directory.
OUTPUT_DIR = "output_dist"

def ensure_output_dir():

    """
    Functionality:
        Creates the output directory for saving plots if it doesn't already exist.

    Args:
        None.

    Returns:
        None.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path="data/drivelm/parsed_nuscenes_driveLM.json"):

    """"
    Functionality:
        Loads the pre-processed NuScenes and DriveLM data from a JSON file.

    Args:
        path (str): The file path to the JSON dataset.

    Returns:
        list: The parsed data as a list of dictionaries.
    """
    with open(path, "r") as f:
        return json.load(f)

# ---------- DriveLM Analysis ----------
def infer_question_type(question: str) -> str:

    """
    Functionality:
        Heuristically assigns a category to a question based on keywords.

    Args:
        question (str): The question string to be categorized.

    Returns:
        str: The inferred question type (e.g., "Object Location",
             "Object Interaction", "Vehicle Behavior", or "Other").
    """

    q = question.lower()
    if "where" in q or "behind" in q or "front" in q or "location" in q:
        return "Object Location"
    elif "interact" in q or "cross" in q or "braking" in q or "moving" in q:
        return "Object Interaction"
    elif "should" in q or "ego" in q or "action" in q or "next" in q:
        return "Vehicle Behavior"
    else:
        return "Other"

def plot_question_types(data):

    """
    Functionality:
        Generates and saves a bar chart of question type distribution.

    Args:
        data (list): The dataset loaded from the JSON file.

    Returns:
        None. A file named `question_type_dist.png` is saved to the output directory.
    """
    types = [infer_question_type(item["question"]) for item in data]
    counter = Counter(types)
    plt.figure(figsize=(6,4))
    plt.bar(counter.keys(), counter.values(), color="green", alpha=0.7)
    plt.title("Question Type Distribution (DriveLM)")
    plt.xlabel("Type")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, "question_types.png"))
    plt.close()

# ---------- NuScenes Analysis ----------
def plot_categories(data):

    """
    Functionality:
        Generates and saves a bar chart of object category distribution.

    Args:
        data (list): The dataset.

    Returns:
        None. A file named `category_dist.png` is saved to the output directory.
    """
    cats = []
    for item in data:
        for ann in item.get("_annotations", []):
            if "category_name" in ann:
                cats.append(ann["category_name"])
    counter = Counter(cats)
    plt.figure(figsize=(10,5))
    plt.bar(counter.keys(), counter.values(), color="orange", alpha=0.7)
    plt.title("Object Category Distribution (NuScenes)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "object_categories.png"))
    plt.close()

def plot_attributes(data):

    attrs = []
    for item in data:
        for ann in item.get("_annotations", []):
            if "attributes" in ann:
                attrs.extend(ann["attributes"])
    counter = Counter(attrs)
    plt.figure(figsize=(10,5))
    plt.bar(counter.keys(), counter.values(), color="purple", alpha=0.7)
    plt.title("Object Attribute Distribution (NuScenes)")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "object_attributes.png"))
    plt.close()

# ---------- Optional: Ego Speed ----------
def compute_speeds(data):

    """
    Functionality:
        Estimates ego vehicle speed based on changes in translation between consecutive frames.

    Args:
        data (list): The dataset.

    Returns:
        list: A list of estimated speeds (relative displacement per frame).
    """

    speeds = []
    last_translation = None
    for item in data:
        translation = item["_annotations"][0]["translation"] if item["_annotations"] else None
        if last_translation and translation:
            dx = translation[0] - last_translation[0]
            dy = translation[1] - last_translation[1]
            dist = (dx**2 + dy**2) ** 0.5
            speeds.append(dist)  # relative displacement
        last_translation = translation
    return speeds

def plot_ego_speed(speeds):

    """
    Functionality:
        Generates and saves a histogram of ego vehicle speeds.

    Args:
        speeds (list): A list of speed values.

    Returns:
        None. A file named `ego_speed_hist.png` is saved to the output directory.
    """

    plt.figure(figsize=(6,4))
    plt.hist(speeds, bins=20, color="blue", alpha=0.7)
    plt.title("Ego Vehicle Speed Distribution")
    plt.xlabel("Relative speed (units/frame)")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(OUTPUT_DIR, "ego_speed_hist.png"))
    plt.close()

# ---------- Main ----------
if __name__ == "__main__":
     
    """
    Main execution block of the script.
    """

    ensure_output_dir()
    data = load_data("data/drivelm/parsed_nuscenes_driveLM.json")  # Change path if needed
    print(f"Loaded {len(data)} samples.")

    # 1. DriveLM Question Type Distribution
    plot_question_types(data)

    # 2. NuScenes Object Category Distribution
    plot_categories(data)

    # 3. NuScenes Object Attribute Distribution
    plot_attributes(data)

    # 4. Optional - Ego Vehicle Speed
    speeds = compute_speeds(data)
    plot_ego_speed(speeds)

    print(f" Analysis complete. Plots saved inside: {OUTPUT_DIR}")
