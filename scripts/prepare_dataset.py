# prepare_dataset.py
# This script prepares and aligns a DriveLM dataset with a NuScenes dataset.

# Standard library imports for file system operations and JSON handling.
import json
from pathlib import Path

# Local imports from a 'parsers' module to handle data parsing.
# NuScenesParser: A class to parse the NuScenes dataset.
# DriveLMParser: A class to parse the raw DriveLM dataset.
# link_question_to_nuscenes: A utility function to link questions to NuScenes frames.
from parsers import NuScenesParser, DriveLMParser, link_question_to_nuscenes

# Third-party imports for data manipulation and progress visualization.
# pandas: A powerful library for data analysis and manipulation.
import pandas as pd
# tqdm: A library to create a smart progress bar for loops.
from tqdm import tqdm

def prepare_and_save_drivelm_nuscenes(nusc_root, drivelm_path, output_json):

    """
    Functionality:
        This function orchestrates the entire data preparation process. It
        initializes the necessary parsers, reads the raw DriveLM questions,
        and then iterates through each question to link it to the corresponding
        NuScenes frame. The aligned data is then compiled into a DataFrame
        and saved as a JSON file.

    Args:
        nusc_root (str): The root directory for the NuScenes dataset.
        drivelm_path (str): The file path for the DriveLM dataset.
        output_json (str): The path where the final aligned JSON dataset will be saved.

    Returns:
        None. The function's primary action is to save the processed data
        to the specified output JSON file.
    """

    print("Initializing NuScenes and DriveLM parsers...")
    print(f"Looking for NuScenes data at: {Path(nusc_root).resolve()}")
    print(f"Looking for DriveLM data at: {Path(drivelm_path).resolve()}")

    if not Path(drivelm_path).exists():
        print(f"Error: DriveLM data file not found at {drivelm_path}. Please check your path.")
        return

    nusc_parser = NuScenesParser(nusc_root)
    
    # This parser is for the raw, un-aligned DriveLM data
    dlm_parser = DriveLMParser(drivelm_path) 
    questions_df = dlm_parser.to_dataframe()
    
    if questions_df.empty:
        print("Warning: No DriveLM questions found. Please check your data path or file content.")
        return

    print("🔹 Aligning DriveLM QAs with NuScenes frames...")
    linked_data = []
    # Use tqdm to show progress for the alignment
    for rec in tqdm(questions_df.to_dict(orient='records'), desc="Aligning data"):
        linked_rec = link_question_to_nuscenes(rec, nusc_parser)
        if linked_rec.get('_linked_sample'):
            linked_data.append(linked_rec)

    output_df = pd.DataFrame(linked_data)
    
    # Save the new, aligned dataset
    output_df.to_json(output_json, orient='records', indent=4)

    print(f"Completed alignment. Successfully linked {len(linked_data)} entries.")
    print(f"Output written to: {output_json}")

if __name__ == '__main__':

    # This standard Python block ensures the following code only runs
    # when the script is executed directly (not when it's imported as a module).

    # Define constants for the file paths. Using uppercase names for constants
    # is a common Python convention.
    # The root directory for the NuScenes-mini dataset.
    # Correct path to the NuScenes-mini folder
    NU_ROOT = 'data/nuscenes/v1.0-mini'
    DRIVELM_PATH = 'data/drivelm/v1_0_train_nus.json'
    OUTPUT_JSON = 'data/drivelm/parsed_nuscenes_driveLM.json'
    
    prepare_and_save_drivelm_nuscenes(NU_ROOT, DRIVELM_PATH, OUTPUT_JSON)