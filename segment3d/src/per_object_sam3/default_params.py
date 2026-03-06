"""
default_params.py - Default parameter values for the per-object SAM3 pipeline.

All tuneable knobs for postsam3_pipeline are centralised here so they can be
imported and referenced consistently by the pipeline function signature, the
CLI argument parser, and any external callers.
"""

from __future__ import annotations

# fmt: off
DEFAULT_PARAMETERS: dict = {
    # ------------------------------------------------------------------
    # Segment-level DBSCAN (2D→3D association noise filtering)
    # Applied per per-frame mask *before* votes are accumulated across frames.
    # ------------------------------------------------------------------

    # Object labels (case-insensitive) whose masks should be excluded from
    # 2D-3D association entirely (e.g. structural elements that are not objects
    # of interest).
    "discard_objects_list": ["wall", "walls", "floor", "ceiling"],

    # Neighbourhood radius in the same units as the COLMAP reconstruction
    # (typically metres).
    "segment_dbscan_eps": 0.5,

    # Minimum number of 3D points required to form a core sample.
    "segment_dbscan_min_samples": 5,
    
    # ------------------------------------------------------------------
    # Mask graph parameters
    # ------------------------------------------------------------------

    # Minimum overlap count (number of shared 3D points) for an edge to be
    # created between two mask nodes in the object mask graph.
    "K": 5,

    # Minimum Jaccard similarity between two masks' 3D point sets for an edge
    # to be kept (range 0–1].
    "tau": 0.8,

    # Minimum number of 3D points a mask node must have to be included in the
    # graph at all.
    "min_points": 1,

    # Minimum number of 3D points a connected component must contain to be
    # reported in the output.
    "min_points_in_3d_segment": 10,

    # ------------------------------------------------------------------
    # Component-level DBSCAN (connected-component cleaning)
    # Applied to the accumulated 3D point cloud for each connected component.
    # ------------------------------------------------------------------

    # Neighbourhood radius in world units.
    "component_dbscan_eps": 1.0,

    # Minimum samples per core point.
    "component_dbscan_min_samples": 5,

    # Discard components that have fewer than this many inlier points after
    # DBSCAN cleaning.
    "component_dbscan_min_points": 20,

    # When True, components with multiple DBSCAN clusters are split into
    # separate components.  When False (default) only noise points are removed.
    "split_components": True,

    # ------------------------------------------------------------------
    # Bounding box parameters
    # ------------------------------------------------------------------

    # Percentile used to clip outlier points before fitting the 3-D bounding
    # box (e.g. 95.0 means the most extreme 5 % of points are ignored).
    "percentile": 95.0,

    # ------------------------------------------------------------------
    # Segment crops parameters
    # ------------------------------------------------------------------

    # Number of top-ranked frames to crop per connected component.
    "top_n": 5,

    # Minimum fraction of the component's 3D points that must project into a
    # frame for that frame to be considered for cropping.
    "min_fraction": 0.05,

    # ------------------------------------------------------------------
    # Captioning parameters
    # ------------------------------------------------------------------

    # Number of top crop images passed to the VLM for generating a caption.
    "caption_n_images": 1,

    # Captioner backend to use (e.g. "vllm").
    "captioner_type": "vllm",

    # HuggingFace model ID for the VLM captioner.
    "caption_model": "Qwen/Qwen2.5-VL-7B-Instruct",

    # GPU device index for captioning inference.
    "caption_device": 0,

    # Batch size for captioning inference.
    "caption_batch_size": 4,

    # ------------------------------------------------------------------
    # CLIP embedding parameters
    # ------------------------------------------------------------------

    # OpenCLIP model name.
    "clip_model": "ViT-H-14",

    # Pretrained weights identifier for the CLIP model.
    "clip_pretrained": "laion2B-s32B-b79K",

    # Batch size for CLIP embedding generation.
    "clip_batch_size": 32,

    # GPU device index for CLIP embedding inference.
    "clip_device": 0,
}
# fmt: on
