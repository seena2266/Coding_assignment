
# drivelm_streamlit.py
# This script creates an interactive Streamlit web application for exploring the
# DriveLM dataset. It features two main tabs:
# 1. Dataset Analysis: Displays statistical charts and allows users to browse
#    the dataset with filters.
# 2. RAG Application: Provides a Retrieval-Augmented Generation interface
#    where users can ask questions about an image and get answers from a
#    combination of a Vision-Language Model (VLM) and a Large Language Model (LLM).

# Imports
# streamlit: The main library for building the web app.
import streamlit as st
# os: For file system operations, like checking if a file exists.
import os
# PIL: Python Imaging Library, used for image manipulation.
from PIL import Image
# traceback: For debugging and displaying error information.
import traceback
# io: For handling byte streams, useful for image data.
import io
# json: For handling JSON data.
import json
# collections: For data structures like Counter and defaultdict.
from collections import Counter, defaultdict

# IMPORTANT: These imports are from your existing code.
# The `search` function is from the `retrieval.py` script and the
# `generate_answer` function is from `generation.py`.
from rag.retrieval import search
from rag.generation import generate_answer

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Project DriveLM Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the path for the data analysis charts
CHART_FOLDER = "output_dist"

# Define Image folder
IMAGE_BASE_DIR = "data/nuscenes"

# Define a consistent size for all charts to ensure a uniform display
UNIFORM_CHART_SIZE = (450, 300)

def combine_and_display_charts(chart_files):

    """
    Functionality:
        Combines a list of chart images into a single vertical image, displays it,
        and provides a download button for the combined image.

    Args:
        chart_files (list): A list of filenames (strings) of the charts to combine.
    """

    if not chart_files:
        st.warning("No charts to combine.")
        return
        
    images = []
    for chart_file in chart_files:
        file_path = os.path.join(CHART_FOLDER, chart_file)
        try:
            image = Image.open(file_path)
            images.append(image.resize(UNIFORM_CHART_SIZE, Image.Resampling.LANCZOS))
        except Exception as e:
            st.error(f"Could not load or resize image file: {chart_file}. Error: {e}")
            return

    # Calculate the total height of the combined image
    total_height = sum(img.size[1] for img in images)
    max_width = max(img.size[0] for img in images)

    # Create a new blank image with a white background
    combined_image = Image.new('RGB', (max_width, total_height), 'white')

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.size[1]
    
    st.subheader("All Charts Combined into a Single Image")
    st.image(combined_image, caption="Combined View of all Charts",  use_container_width=True)

    # Create a download button for the combined image
    try:
        buf = io.BytesIO()
        combined_image.save(buf, format="PNG")
        st.download_button(
            label="Download Combined Charts as PNG",
            data=buf.getvalue(),
            file_name="drive_lm_dashboard_charts.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Failed to create download button: {e}")

# --- Function to display the Data Analysis tab ---
def display_data_analysis_dashboard():

    """
    Renders the data analysis charts from the specified folder.
    This function checks if the `output_dist` folder exists and
    displays all image files found within it in a two-column layout.
    """

    st.markdown("<h2 style='color: #1E3A8A;'>📊 DriveLM Dataset Distribution Analysis</h2>", 
                unsafe_allow_html=True)
    st.markdown("##### This dashboard visualizes key distributions and attributes of the DriveLM "
                "and NuScenes-mini datasets. The charts below provide insights into the different " \
                "question types, object categories, and vehicle states.")

    if not os.path.exists(CHART_FOLDER):
        st.error(f"Error: The directory '{CHART_FOLDER}' was not found.")
        st.markdown(
            "Please ensure your generated charts are saved in a folder named " \
            "`output_dist` in the same directory as this Streamlit app."
        )
        return

    # List all image files in the charts folder
    chart_files = [f for f in os.listdir(CHART_FOLDER) if f.lower().endswith(('.png', '.jpg', 
                                                                              '.jpeg', '.gif'))]
    chart_files.sort()  # Sort files for a consistent display order

    if not chart_files:
        st.warning(f"No charts found in the directory '{CHART_FOLDER}'.")
        st.markdown(
            "Please ensure your data analysis script has successfully generated and saved the " \
            "charts in the specified folder."
        )
        return

    # Use a checkbox to toggle between individual charts and a single combined image
    show_combined = st.checkbox("Show Combined Charts", value=False)
    
    if show_combined:
        combine_and_display_charts(chart_files)
    else:
        # Use a two-column layout for a clean, side-by-side view of the charts
        col1, col2 = st.columns(2)
        for i, chart_file in enumerate(chart_files):
            # Alternate between columns to arrange images neatly
            target_column = col1 if i % 2 == 0 else col2
            
            with target_column:
                file_path = os.path.join(CHART_FOLDER, chart_file)
                try:
                    # Open the image and resize it to a uniform size
                    image = Image.open(file_path)
                    resized_image = image.resize(UNIFORM_CHART_SIZE, Image.Resampling.LANCZOS)
                    
                    # Display the resized image with a descriptive caption
                    st.image(resized_image, caption=f"Chart: {chart_file}",  use_container_width=True)
                    st.markdown("<h3 style='color: green;'>***********************************"
                                "**************</h3>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Could not load or resize image file: {chart_file}. Error: {e}")
                    st.warning("Please check the file integrity and permissions.")

# --- Function to display the RAG Application tab ---
def display_rag_application():

    """
    Renders the RAG application logic provided by the user.
    This includes the query input, search button, and display of results.
    """

    st.title("🚗 Retrieval-Augmented QA for DriveLM + NuScenes")
    st.markdown("Ask a question and get a context-rich, generated answer.")
    
    # Text input for the user's query
    user_query = st.text_input("Ask a question:", 
                               placeholder="e.g., What is the vehicle's speed and position?")

    # The main logic is triggered when the "Search" button is clicked
    if st.button("Search") and user_query.strip():
        try:
            with st.spinner("Retrieving context from the dataset..."):
                retrieved_context = search(user_query)

            if retrieved_context:
                st.subheader("🔍 Retrieved Context")
                st.markdown(f"**Question:** {retrieved_context.get('question')}")
                st.markdown(f"**Answer:** {retrieved_context.get('answer')}")

                st.subheader("📊 Structured Data")
                st.json({
                    "attributes": retrieved_context.get("attributes", []),
                    "linked_sample": retrieved_context.get("linked_sample", {}),
                    "ego_pose": retrieved_context.get("ego_pose", {})
                })

                if retrieved_context.get("resized_image"):
                    st.image(retrieved_context["resized_image"], 
                             caption="Retrieved and Resized NuScenes Image")
                else:
                    st.warning("No image available.")

                st.subheader("🤖 Generated Answer")
                with st.spinner("Generating answer..."):
                    final_answer = generate_answer(user_query, retrieved_context)
                st.markdown(final_answer)
            else:
                st.error("No results found. Please try a different query.")
        except Exception as e:
            st.error("An error occurred during the retrieval or generation process.")
            st.error(f"Error details: {e}")
            st.code(traceback.format_exc())
# ----------------------------
# Helper functions
# ----------------------------
def load_samples(json_path):

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Expect either a list of samples or a dict with key 'data' / 'samples'; try to normalize
    if isinstance(data, dict):
        # try common keys
        for k in ("samples", "data", "items"):
            if k in data and isinstance(data[k], list):
                return data[k]
        # fallback: wrap single sample
        return [data]
    return data


def compute_global_counts(samples):
    cat_counter = Counter()
    attr_counter = Counter()
    qtype_counter = Counter()
    scene_counter = Counter()

    for s in samples:
        q = s.get("question", "")
        # if you have an explicit question type field, use it; else try to 
        # infer from question text (optional)
        qtype = s.get("question_type") or s.get("type") or "unknown"
        qtype_counter[qtype] += 1

        anns = s.get("_annotations", [])
        for a in anns:
            cat = a.get("category_name", "unknown")
            cat_counter[cat] += 1
            for attr in a.get("attributes", []):
                attr_counter[attr] += 1

        scene_token = s.get("scene_token")
        if scene_token:
            scene_counter[scene_token] += 1

    return cat_counter, attr_counter, qtype_counter, scene_counter


def pick_interesting_samples(
    samples,
    rare_categories,
    min_annotations=2,
    prefer_interaction_question=False,
    top_k=5,
):
    """
    Score samples by:
      + 3 points if any annotation belongs to rare_categories
      + +1 for each annotation beyond min_annotations
      + +2 if question_type is interaction (if available)
      + +1 if contains uncommon attributes (optional)
    """
    scored = []
    for i, s in enumerate(samples):
        score = 0
        anns = s.get("_annotations", [])
        cats = [a.get("category_name", "") for a in anns]
        attrs = [a.get("attributes", []) for a in anns]

        # rare category boost
        if any(c in rare_categories for c in cats):
            score += 3

        # multiple objects boost
        if len(anns) >= min_annotations:
            score += (len(anns) - 1)

        # prefer interaction-type questions (if there is a field)
        qtype = s.get("question_type") or s.get("type") or ""
        if prefer_interaction_question and "interaction" in qtype.lower():
            score += 2

        # uncommon attribute heuristic
        flat_attrs = [a for sub in attrs for a in sub]
        # small boost if attributes show mixed actor states
        if "pedestrian.moving" in flat_attrs and "vehicle.moving" in flat_attrs:
            score += 2

        if score > 0:
            scored.append((score, i, s))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [entry for _, _, entry in scored[:top_k]]

# --- Main Application Logic ---
def main():
     
    """
    Functionality:
        The main function that runs the Streamlit application.
        It creates the tabs and loads the data, ensuring it is ready for both
        the analysis dashboard and the RAG application.
    """
    st.header("Project DriveLM - Final Deliverable")

    # Use tabs to organize the app into logical sections
    tab1, tab2 = st.tabs(["📊 Data Analysis", "🤖 RAG Application"])

    with tab1:

        display_data_analysis_dashboard()
        # Report content as a static markdown block
        report_content = """
        # Identifying Patterns & Anomalies

        ### Findings (Patterns & Anomalies)
        ### Question type distribution

        * **Object Location** questions dominate by a large margin — far 
        more than Vehicle Behavior or Object Interaction.
        * **Pattern:** The dataset has a strong emphasis on localization 
        (where things are relative to ego).
        * **Implication/bias:** Models trained on this dataset may become 
        strong at localization tasks but weak on interaction reasoning 
        (because Object Interaction is rare).
        * **Possible anomaly to check:** Are some interaction questions 
        omitted or mis-labeled as "Other" or "Object Location"?

        ### Object category distribution

        * **Pedestrians** and **cars** are the most common categories 
        (these two categories far outnumber others).
        * Some object types (motorcycle, trailer, bicycle, truck, police officer) 
        are very underrepresented.
        * **Pattern:** The dataset has a strong urban driving focus with 
        pedestrians and cars; other road actors are sparse.
        * **Implication/bias:** The model will generalize poorly to 
        rare classes (motorcycles, trailers, emergency vehicles).
        * A non-trivial count for `movable_object.debris` / `movable_object.barrier` 
        (if present) suggests some samples contain obstacles—this is good, 
        but still likely less frequent than cars/pedestrians.
        * **Action:** Calculate per-class frequency and set a threshold 
        (e.g., `<1%` of all objects) to define "rare classes."

        ### Object attribute distribution

        * **Moving pedestrians** is the single biggest attribute group; 
        vehicle moving/parked/stopped are well represented too.
        * **Pattern:** Many dynamic object states are present (moving vs parked vs stopped).
        * **Implication:** The dataset will likely teach models better on dynamic 
        state detection for pedestrians and cars, but cycling states are rare 
        (`cycle_with_rider`, `cycle_without_rider` are small).
        * **Action:** Check the co-occurrence of attributes and classes 
        (e.g., how many `pedestrian.moving` co-occur with object interaction questions).

        ### High-level anomalies & checks

        * **Underrepresentation of Object Interaction** relative to 
        object counts—many annotated objects but few interaction questions. Investigate why:
        * Are interaction questions not generated?
        * Are interactions labeled as other types?
        * **Scene imbalance risk:** Verify if few scenes contribute a 
        disproportionate number of questions/annotations (scene-level long-tail).
        * **Missing data issues to check:** Samples with `num_lidar_pts == 0` or 
        `num_radar_pts == 0` may indicate occluded or potentially noisy annotations. 
        Check image existence for all `_linked_sample_data.filename`.
        * **Temporal imbalance:** Check timestamps to ensure data is not 
        heavily concentrated in a few time periods.
        """
        st.markdown(report_content, unsafe_allow_html=True)
        st.markdown("<h3 style='color: green;'>****************************************"
        "****************************************</h3>", unsafe_allow_html=True)

        SAMPLES_JSON = "data/drivelm/parsed_nuscenes_driveLM.json"
        # Load samples JSON
        if not os.path.exists(SAMPLES_JSON):
            st.error(
            f"Samples JSON file not found at '{SAMPLES_JSON}'. Please provide your samples file path."
            )
        else:
            samples = load_samples(SAMPLES_JSON)
            cat_counter, attr_counter, qtype_counter, scene_counter = compute_global_counts(samples)
            st.subheader("Global counts (quick)")

            # show top categories and attributes
            top_cats = cat_counter.most_common(10)
            top_attrs = attr_counter.most_common(10)
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("Top object categories (top 10)")
                for k, v in top_cats:
                    st.write(f"- `{k}` : {v}")
            with col_b:
                st.write("Top attributes (top 10)")
                for k, v in top_attrs:
                    st.write(f"- `{k}` : {v}")

            # allow user to set rare threshold
            st.sidebar.header("Sample selection controls")
            rare_frac = st.sidebar.slider(
                "Define rare category threshold (fraction of total objects)",
                min_value=0.0005,
                max_value=0.05,
                value=0.01,
                step=0.0005,
            )
            total_objects = sum(cat_counter.values()) if sum(cat_counter.values()) > 0 else 1
            # determine rare categories
            rare_categories = [
                cat for cat, cnt in cat_counter.items() if (cnt / total_objects) < rare_frac
            ]
            st.sidebar.write(f"Detected {len(rare_categories)} rare categories (threshold {rare_frac})")

            # sample selection options
            min_ann = st.sidebar.slider("Min annotations to qualify as 'complex'", 1, 8, 2)
            prefer_interaction = st.sidebar.checkbox("Prefer Object Interaction question samples", 
                                                     value=True)
            max_samples = st.sidebar.slider("Max samples to show", 1, 8, 5)

            # pick interesting samples
            interesting = pick_interesting_samples(
                samples,
                rare_categories=rare_categories,
                min_annotations=min_ann,
                prefer_interaction_question=prefer_interaction,
                top_k=max_samples,
            )
            st.markdown("<h3 style='color: green;'>********************************************"
            "*********************</h3>", unsafe_allow_html=True)

            st.subheader("Interesting / challenging samples (selected)")
            if not interesting:
                st.info("No samples matched the selection heuristics. Try loosening the threshold.")
            else:
                for s in interesting:
                    image_rel = None
                    q = s.get("question", "N/A")
                    ans = s.get("answer", "N/A")
                    anns = s.get("_annotations", [])
                    

                    image_rel = s.get("_linked_sample_data", {}).get("filename", None)
                    # build absolute image path
                    image_path = None
                    if image_rel:
                        image_path = os.path.join(IMAGE_BASE_DIR, image_rel)  
                    image_path = image_path.replace("\\", "/")  
                    # metadata: categories and attributes summary
                    cats = [a.get("category_name", "") for a in anns]
                    attrs = [a.get("attributes", []) for a in anns]
                    flat_attrs = [it for sub in attrs for it in sub]
                    card_cols = st.columns([1, 1.2])
                    with card_cols[0]:
                        if image_path and os.path.exists(image_path):
                            st.image(image_path, use_container_width=True)
                        else:
                            st.write("Image not found or path invalid:")
                            st.code(str(image_path))
                    with card_cols[1]:
                        st.markdown("### Question")
                        st.write(q)
                        st.markdown("### Answer (if present)")
                        st.write(ans)
                        st.markdown("**Objects in annotation**")
                        st.write(", ".join(cats) if cats else "No annotations found")
                        st.markdown("**Attributes**")
                        st.write(", ".join(flat_attrs) if flat_attrs else "No attributes")
                        # show a small JSON snippet for debugging
                        with st.expander("Show raw annotation JSON"):
                            st.json(anns)
                    st.markdown("---")

            st.info(
            "Use the sidebar to change the rarity threshold and filters. For reproducibility, " \
            "compute global counts once and save them to disk."
            )

    with tab2:
        display_rag_application()

if __name__ == "__main__":
    main()
