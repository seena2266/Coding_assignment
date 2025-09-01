# parsers.py
# This file provides a collection of utility functions and parsers for
# processing and linking data from the NuScenes and DriveLM datasets.

# Standard library imports for handling file paths, JSON, and types.
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
import math


# ---------------------------
# Utility
# ---------------------------
def load_json(path: Path) -> Any:

    """
    Load JSON from a given file path.

    This function attempts to load and parse a JSON file. It accepts
    file paths as either `Path` objects or strings.

    Args:
        path (Path): The file path to the JSON file.

    Returns:
        Any: The parsed JSON data (a dictionary, list, etc.).

    Raises:
        FileNotFoundError: If the specified JSON file does not exist.
        Exception: If there is an error parsing the JSON content.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# NuScenes Parser (enriched)
# ---------------------------------------------------------------------
class NuScenesParser:

    """
    Parser for NuScenes metadata JSONs.
    Loads sample, sample_data, sample_annotation, instance, category, ego_pose, etc.
    Provides helpers to get annotations enriched with category_name and attributes,
    and ego-pose helpers.
    """

    def __init__(self, root: str):

        """
        Initializes the parser by loading all relevant NuScenes JSON files into DataFrames.

        Args:
            root (str): The root directory for the NuScenes dataset.

        Raises:
            FileNotFoundError: If the specified NuScenes root directory does not exist.
        """

        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"NuScenes root not found: {self.root}")

        # Core JSON files (try to load, but allow missing files by using empty DataFrame)
        def _safe_load(fname: str):
            p = self.root / fname
            if p.exists():
                try:
                    return pd.DataFrame(load_json(p))
                except Exception:
                    return pd.DataFrame([])
            else:
                return pd.DataFrame([])

        self.samples = _safe_load("sample.json")
        self.sample_data = _safe_load("sample_data.json")
        self.annotations = _safe_load("sample_annotation.json")
        self.instances = _safe_load("instance.json")
        self.categories = _safe_load("category.json")
        self.ego_poses = _safe_load("ego_pose.json")

        # Optional JSONs
        self.scene = _safe_load("scene.json")
        self.sensor = _safe_load("sensor.json")
        self.attributes = _safe_load("attribute.json")

        # Build quick lookup maps where possible
        try:
            self.category_map = dict(zip(self.categories["token"], self.categories["name"]))
        except Exception:
            self.category_map = {}

        try:
            self.instance_map = dict(zip(self.instances["token"], self.instances["category_token"]))
        except Exception:
            self.instance_map = {}

    # --------------------------
    # Basic accessors (preserve old API)
    # --------------------------
    def get_sample_by_token(self, token: str) -> Optional[Dict]:

        """
        Retrieves a sample record by its unique token.

        Args:
            token (str): The unique token for the sample.

        Returns:
            Optional[Dict]: A dictionary representing the sample record, or None if not found.
        """
        if self.samples.empty:
            return None
        matches = self.samples[self.samples.get("token") == token].to_dict("records")
        return matches[0] if matches else None

    def get_sample_data(self, token: str) -> Optional[Dict]:

        """
        Retrieves a sample data record by its unique token.

        Args:
            token (str): The unique token for the sample data.

        Returns:
            Optional[Dict]: A dictionary representing the sample data record, or None if not found.
        """

        if self.sample_data.empty:
            return None
        matches = self.sample_data[self.sample_data.get("token") == token].to_dict("records")
        return matches[0] if matches else None

    # --------------------------
    # Instance / attribute helpers
    # --------------------------
    def get_instance_by_token(self, instance_token: str) -> Optional[Dict]:

        """
        Retrieves an instance record by its unique token.

        Args:
            instance_token (str): The unique token for the instance.

        Returns:
            Optional[Dict]: A dictionary representing the instance record, or None if not found.
        """
         
        if self.instances.empty:
            return None
        matches = self.instances[self.instances.get("token") == instance_token].to_dict("records")
        return matches[0] if matches else None

    def get_attribute_names(self, attribute_tokens: List[str]) -> List[str]:

        """
        Maps a list of attribute tokens to their corresponding readable names.

        Args:
            attribute_tokens (List[str]): A list of attribute tokens.

        Returns:
            List[str]: A list of readable attribute names. Returns an empty list if
                       `attributes.json` is not present or if the input list is empty.
        """

        if not attribute_tokens or self.attributes.empty:
            return []
        if not hasattr(self, "_attr_token_map"):
            try:
                self._attr_token_map = dict(zip(self.attributes["token"], self.attributes["name"]))
            except Exception:
                self._attr_token_map = {}
        return [self._attr_token_map.get(t, t) for t in attribute_tokens]

    # --------------------------
    # Annotation enrichment (keeps old behavior + adds category/attributes)
    # --------------------------
    def get_annotations_for_sample(self, sample_token: str) -> List[Dict]:

        """
        Returns a list of annotation dictionaries for a given sample token.

        Each annotation is enriched with `category_token_instance` (from `instance.json`),
        `category_name` (from `category.json`), and readable `attributes` (from `attribute.json`).

        Args:
            sample_token (str): The unique token of the sample.

        Returns:
            List[Dict]: A list of enriched annotation dictionaries, or an empty list if no
                        annotations are found.
        """

        if self.annotations.empty:
            return []

        anns_df = self.annotations[self.annotations.get("sample_token") == sample_token]
        if anns_df.empty:
            return []

        records = anns_df.to_dict("records")
        enriched = []
        for ann in records:
            inst_token = ann.get("instance_token")
            cat_token = None
            cat_name = None
            if inst_token:
                inst = self.get_instance_by_token(inst_token)
                if inst:
                    cat_token = inst.get("category_token")
                    # Try the category_map first
                    cat_name = self.category_map.get(cat_token) if cat_token else None
                    # fallback: try to find in categories dataframe
                    if cat_name is None and not self.categories.empty:
                        try:
                            cat_row = self.categories[self.categories.get("token") == cat_token]
                            if not cat_row.empty:
                                cat_name = cat_row.iloc[0].get("name")
                        except Exception:
                            cat_name = None
            ann["category_token_instance"] = cat_token
            ann["category_name"] = cat_name
            # attributes
            ann["attributes"] = self.get_attribute_names(ann.get("attribute_tokens") or [])
            enriched.append(ann)
        return enriched

    # --------------------------
    # Ego pose helpers
    # --------------------------
    def get_ego_pose_by_token(self, ego_token: str) -> Optional[Dict]:
        
        """
        Retrieves an ego pose record by its unique token.

        Args:
            ego_token (str): The unique token for the ego pose.

        Returns:
            Optional[Dict]: A dictionary representing the ego pose record, or None if not found.
        """

        if ego_token is None or self.ego_poses.empty:
            return None
        matches = self.ego_poses[self.ego_poses.get("token") == ego_token].to_dict("records")
        return matches[0] if matches else None

    def get_ego_translation_for_sample(self, sample_token: str) -> Optional[Dict]:
       
        """
        Retrieves the ego vehicle's translation, rotation, and timestamp for a given sample.

        This function looks for a key frame within the `sample_data` entries for the given
        `sample_token` to find the corresponding `ego_pose_token`, and then retrieves the
        full pose from the `ego_pose` DataFrame.

        Args:
            sample_token (str): The unique token of the sample.

        Returns:
            Optional[Dict]: A dictionary containing the 'translation', 'rotation',
                            and 'timestamp' of the ego vehicle's pose, or None if not found.
        """
        if self.sample_data.empty:
            return None
        sdf = self.sample_data[self.sample_data.get("sample_token") == sample_token]
        if sdf.empty:
            return None

        # prefer key_frame rows
        try:
            rec_df = sdf[sdf.get("is_key_frame") == True]
            if rec_df.empty:
                rec_df = sdf.iloc[[0]]
            else:
                rec_df = rec_df.iloc[[0]]
            rec = rec_df.to_dict("records")[0]
        except Exception:
            try:
                rec = sdf.iloc[0].to_dict()
            except Exception:
                return None

        ego_token = rec.get("ego_pose_token") or rec.get("ego_pose")
        if not ego_token:
            return None
        ego = self.get_ego_pose_by_token(ego_token)
        if not ego:
            return None
        return {
            "translation": ego.get("translation"),
            "rotation": ego.get("rotation"),
            "timestamp": ego.get("timestamp")
        }

    def estimate_ego_speed_between_samples(self, 
                                           sample_token_a: str, sample_token_b: str) -> Optional[float]:
       
        """
        Estimates the ego vehicle's speed between two samples in meters per second.

        Calculates the Euclidean distance between the translations of the two samples'
        ego poses and divides it by the time difference.

        Args:
            sample_token_a (str): The unique token of the first sample.
            sample_token_b (str): The unique token of the second sample.

        Returns:
            Optional[float]: The estimated speed in m/s, or None if the speed cannot be
                             computed (e.g., if data is missing or time difference is zero).
        """
        a = self.get_ego_translation_for_sample(sample_token_a)
        b = self.get_ego_translation_for_sample(sample_token_b)
        if not a or not b:
            return None
        ta = a.get("timestamp")
        tb = b.get("timestamp")
        ta_trans = a.get("translation")
        tb_trans = b.get("translation")
        if ta is None or tb is None or ta_trans is None or tb_trans is None:
            return None
        try:
            dt = float(tb - ta)
            # convert microseconds -> seconds if necessary
            if dt > 1e6:
                dt = dt / 1e6
            if dt == 0:
                return None
        except Exception:
            return None
        xa, ya, za = ta_trans
        xb, yb, zb = tb_trans
        dist = math.sqrt((xb - xa) ** 2 + (yb - ya) ** 2 + (zb - za) ** 2)
        return dist / dt if dt != 0 else None


# ---------------------------------------------------------------------
# DriveLM Parser (robust; preserves old flattening behavior)
# ---------------------------------------------------------------------
class DriveLMParser:
    """
    Parser for DriveLM question-answer data.
    This implementation tries multiple formats:
      - old dict: { scene_token: { "key_frames": { sample_token: { "QA": {...} } } } }
      - list of scene dicts: [ { "scene_token": ..., 
        "key_frames": [ { "sample_token": ..., "qas":[...] } ] } ]
      - other small variants
    It extracts question/answer/reasoning into a flat DataFrame with columns:
      scene_token, sample_token, question, answer, reasoning
    """

    def __init__(self, drivelm_json: str):

        """
        Initializes the parser by loading the DriveLM JSON file.

        Args:
            drivelm_json (str): The file path to the DriveLM dataset.

        Raises:
            FileNotFoundError: If the specified DriveLM JSON file does not exist.
        """
        self.path = Path(drivelm_json)
        if not self.path.exists():
            raise FileNotFoundError(f"DriveLM JSON not found: {self.path}")
        self.data = load_json(self.path)

    @staticmethod
    def _extract_qa_from_pair(pair: Dict) -> Dict:

        """
        Normalizes a question-answer pair dictionary by checking various
        possible key names for question, answer, and reasoning.

        Args:
            pair (Dict): A dictionary representing a single QA pair.

        Returns:
            Dict: A normalized dictionary with 'question', 'answer', and
                  'reasoning' keys.
        """

        if not isinstance(pair, dict):
            return {"question": None, "answer": None, "reasoning": None}
        q = pair.get("Q") or pair.get("question") or pair.get("q") or pair.get("Question")
        a = pair.get("A") or pair.get("answer") or pair.get("a") or pair.get("Answer")
        c = pair.get("C") or pair.get("reasoning") or pair.get(
            "chain_of_thought") or pair.get("explanation")
        return {"question": q, "answer": a, "reasoning": c}

    def to_dataframe(self) -> pd.DataFrame:

        """
        Flattens the DriveLM data into a pandas DataFrame.

        This method handles different top-level JSON structures (list or dictionary)
        and extracts all question-answer pairs into a flat DataFrame with
        `scene_token`, `sample_token`, `question`, `answer`, and `reasoning` columns.

        Returns:
            pd.DataFrame: A DataFrame containing the flattened DriveLM data.
        """
        records = []

        # Case A: top-level is a dict keyed by scene_token (old DriveLM format)
        if isinstance(self.data, dict):
            for scene_token, scene_data in self.data.items():
                if not isinstance(scene_data, dict):
                    continue
                key_frames = scene_data.get("key_frames") or scene_data.get("key_frames_map") or {}
                # key_frames could be a dict mapping sample_token->sample_data
                if isinstance(key_frames, dict):
                    for sample_token, sample_data in key_frames.items():
                        if not isinstance(sample_data, dict):
                            continue
                        # QA might be under 'QA' with categories mapping to lists
                        if "QA" in sample_data:
                            qa_obj = sample_data.get("QA", {})
                            if isinstance(qa_obj, dict):
                                for qa_list in qa_obj.values():
                                    if isinstance(qa_list, list):
                                        for qa_pair in qa_list:
                                            norm = self._extract_qa_from_pair(qa_pair)
                                            records.append({
                                                "scene_token": scene_token,
                                                "sample_token": sample_token,
                                                "question": norm["question"],
                                                "answer": norm["answer"],
                                                "reasoning": norm["reasoning"]
                                            })
                        # Old variants or lowercase 'qas'
                        elif "qas" in sample_data and isinstance(sample_data.get("qas"), list):
                            for qa_pair in sample_data.get("qas", []):
                                norm = self._extract_qa_from_pair(qa_pair)
                                records.append({
                                    "scene_token": scene_token,
                                    "sample_token": sample_token,
                                    "question": norm["question"],
                                    "answer": norm["answer"],
                                    "reasoning": norm["reasoning"]
                                })
                # key_frames could be a list of frames
                elif isinstance(key_frames, list):
                    for frame in key_frames:
                        if not isinstance(frame, dict):
                            continue
                        sample_token = frame.get("sample_token") or frame.get("sampleToken")
                        if not sample_token:
                            continue
                        # qas in frame
                        if "qas" in frame and isinstance(frame.get("qas"), list):
                            for qa_pair in frame.get("qas", []):
                                norm = self._extract_qa_from_pair(qa_pair)
                                records.append({
                                    "scene_token": scene_token,
                                    "sample_token": sample_token,
                                    "question": norm["question"],
                                    "answer": norm["answer"],
                                    "reasoning": norm["reasoning"]
                                })
                        elif "QA" in frame and isinstance(frame.get("QA"), dict):
                            qa_obj = frame.get("QA")
                            for qa_list in qa_obj.values():
                                if isinstance(qa_list, list):
                                    for qa_pair in qa_list:
                                        norm = self._extract_qa_from_pair(qa_pair)
                                        records.append({
                                            "scene_token": scene_token,
                                            "sample_token": sample_token,
                                            "question": norm["question"],
                                            "answer": norm["answer"],
                                            "reasoning": norm["reasoning"]
                                        })

        # Case B: top-level is a list (common alternative format)
        elif isinstance(self.data, list):
            for scene in self.data:
                if not isinstance(scene, dict):
                    # skip strings or other non-dict entries
                    continue
                scene_token = scene.get("scene_token") or scene.get("sceneToken") or scene.get("token")
                key_frames = scene.get("key_frames") or scene.get("frames") or scene.get("keyFrames", [])
                # key_frames sometimes is a list
                if isinstance(key_frames, list):
                    for frame in key_frames:
                        if not isinstance(frame, dict):
                            continue
                        sample_token = frame.get("sample_token") or frame.get(
                            "sampleToken") or frame.get("sample")
                        # qas may be under 'qas' (list) or 'QA' (dict)
                        if "qas" in frame and isinstance(frame.get("qas"), list):
                            for qa_pair in frame.get("qas", []):
                                norm = self._extract_qa_from_pair(qa_pair)
                                records.append({
                                    "scene_token": scene_token,
                                    "sample_token": sample_token,
                                    "question": norm["question"],
                                    "answer": norm["answer"],
                                    "reasoning": norm["reasoning"]
                                })
                        elif "QA" in frame and isinstance(frame.get("QA"), dict):
                            qa_obj = frame.get("QA")
                            for qa_list in qa_obj.values():
                                if isinstance(qa_list, list):
                                    for qa_pair in qa_list:
                                        norm = self._extract_qa_from_pair(qa_pair)
                                        records.append({
                                            "scene_token": scene_token,
                                            "sample_token": sample_token,
                                            "question": norm["question"],
                                            "answer": norm["answer"],
                                            "reasoning": norm["reasoning"]
                                        })
                # key_frames could be a dict mapping sample_token -> sample_data (rare in list form)
                elif isinstance(key_frames, dict):
                    for sample_token, sample_data in key_frames.items():
                        if not isinstance(sample_data, dict):
                            continue
                        if "qas" in sample_data and isinstance(sample_data.get("qas"), list):
                            for qa_pair in sample_data.get("qas", []):
                                norm = self._extract_qa_from_pair(qa_pair)
                                records.append({
                                    "scene_token": scene_token,
                                    "sample_token": sample_token,
                                    "question": norm["question"],
                                    "answer": norm["answer"],
                                    "reasoning": norm["reasoning"]
                                })
                        elif "QA" in sample_data and isinstance(sample_data.get("QA"), dict):
                            qa_obj = sample_data.get("QA")
                            for qa_list in qa_obj.values():
                                if isinstance(qa_list, list):
                                    for qa_pair in qa_list:
                                        norm = self._extract_qa_from_pair(qa_pair)
                                        records.append({
                                            "scene_token": scene_token,
                                            "sample_token": sample_token,
                                            "question": norm["question"],
                                            "answer": norm["answer"],
                                            "reasoning": norm["reasoning"]
                                        })

        # Normalize into DataFrame
        return pd.DataFrame(records)


# ---------------------------------------------------------------------
# Linking function (preserve old per-question linking behavior)
# ---------------------------------------------------------------------
def link_question_to_nuscenes(question_record: Dict, nuscenes: NuScenesParser) -> Dict:

    """
    Link a single DriveLM QA record (dict) to NuScenes metadata.

    This function enriches a DriveLM question record with relevant NuScenes data.
    It attaches the corresponding `_linked_sample_data` (specifically, for the
    front camera if a key frame is available), the full `_linked_sample` record,
    `_annotations` for that sample, and the `_ego_pose` of the vehicle.

    Args:
        question_record (Dict): A single dictionary representing a DriveLM
                                question-answer-reasoning entry.
        nuscenes (NuScenesParser): An instance of the NuScenesParser class.

    Returns:
        Dict: The original `question_record` enriched with NuScenes metadata.
              Returns the original record unchanged if the sample token is missing.
    """
    out = question_record.copy()
    out["_annotations"] = []
    out["_linked_sample"] = None
    out["_linked_sample_data"] = None
    out["_ego_pose"] = None

    sample_token = question_record.get("sample_token") or question_record.get(
        "sample") or question_record.get("sampleToken")
    if not sample_token:
        return out

    # sample_data selection: prefer key_frame & jpg
    if not nuscenes.sample_data.empty:
        try:
            cam_front_data = nuscenes.sample_data[
                (nuscenes.sample_data.get("sample_token") == sample_token) &
                (nuscenes.sample_data.get("is_key_frame") == True) &
                (nuscenes.sample_data.get("fileformat") == "jpg")
            ].to_dict("records")
        except Exception:
            cam_front_data = nuscenes.sample_data[nuscenes.sample_data.get(
                "sample_token") == sample_token].to_dict("records")

        if cam_front_data:
            # pick first record
            out["_linked_sample_data"] = cam_front_data[0]

    # sample record
    sample = nuscenes.get_sample_by_token(sample_token)
    if sample:
        out["_linked_sample"] = sample
        out["_annotations"] = nuscenes.get_annotations_for_sample(sample_token)
        out["_ego_pose"] = nuscenes.get_ego_translation_for_sample(sample_token)

    return out
