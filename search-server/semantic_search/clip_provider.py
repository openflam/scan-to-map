"""CLIP-based semantic search provider using FAISS."""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import faiss
import torch
import open_clip

from .base import SemanticSearchProvider


class CLIPProvider(SemanticSearchProvider):
    """Semantic search provider using CLIP embeddings and FAISS index."""

    def __init__(
        self,
        faiss_index_path: str,
        embeddings_npz_path: str,
        component_captions: Dict[int, Any],
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        top_k: int = 10,
        gap_threshold: float = 0.05,
        device: Optional[str] = None,
    ):
        """
        Initialize the CLIP provider with FAISS index and CLIP model.

        Args:
            faiss_index_path: Path to the FAISS index file
            embeddings_npz_path: Path to the .npz file containing component_ids
            component_captions: Dictionary keyed by component ID with captions
            model_name: CLIP model name (default: ViT-B-32)
            pretrained: Pretrained weights (default: laion2b_s34b_b79k)
            top_k: Number of top matching components to return
            gap_threshold: Minimum gap in similarity scores for elbow detection
            ratio_threshold: Minimum ratio of current/previous score for elbow detection
            device: Device to use for CLIP model (None for auto-detect)
        """
        self.top_k = top_k
        self.gap_threshold = gap_threshold
        self.component_captions = component_captions

        # Set up device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading CLIP model: {model_name} ({pretrained}) on {self.device}...")

        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Load FAISS index
        print(f"Loading FAISS index from: {faiss_index_path}...")
        self.index = faiss.read_index(str(faiss_index_path))

        # Load component IDs mapping
        print(f"Loading component IDs from: {embeddings_npz_path}...")
        data = np.load(embeddings_npz_path)
        self.component_ids = data["component_ids"]

        # Convert component_ids to integers for consistency
        self.component_ids = [int(cid) for cid in self.component_ids]

        print(f"CLIP provider initialized with {len(self.component_ids)} components")

    def _encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query using CLIP text encoder.

        Args:
            text: Text query to encode

        Returns:
            Normalized embedding as numpy array
        """
        # Tokenize text
        text_tokens = self.tokenizer([text])
        text_tokens = text_tokens.to(self.device)

        # Encode text
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # Normalize embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Convert to numpy and ensure float32
        return text_features.cpu().numpy().astype(np.float32)

    def match_components(
        self, query: str, component_captions: Optional[Dict[int, Any]] = None
    ) -> Dict[str, Any]:
        """
        Match a search query to components using CLIP embeddings.

        Args:
            query: The search query string
            component_captions: Not used (component captions are provided at initialization)

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice with captions
        """

        # Encode the query text
        query_embedding = self._encode_text(query)

        # Search in FAISS index
        # Note: FAISS uses L2 distance by default for IndexHNSWFlat
        # For normalized vectors, L2 distance is related to cosine similarity
        distances, indices = self.index.search(query_embedding, self.top_k)

        # Convert L2 distances to cosine similarities
        # For normalized vectors: cosine_sim = 1 - (L2_distance^2 / 2)
        similarities = 1 - (distances[0] ** 2) / 2

        keep = list(range(len(similarities)))  # Keep all results for now

        # Apply elbow logic to determine which results to keep
        keep = [0]  # Always keep the top result

        for i in range(1, len(similarities)):
            if i >= len(similarities):
                break

            prev_score = similarities[i - 1]
            cur_score = similarities[i]

            # Apply elbow detection: stop if there's a significant gap
            gap = prev_score - cur_score

            if gap >= self.gap_threshold:
                break

            keep.append(i)

        # Extract component IDs and build reason with captions
        matched_component_ids = []
        caption_details = []

        for i in keep:
            idx = indices[0, i]
            comp_id = self.component_ids[idx]
            similarity = similarities[i]

            matched_component_ids.append(comp_id)

            # Get caption if available
            caption = "No caption available"
            if comp_id in self.component_captions:
                caption = self.component_captions[comp_id].get(
                    "caption", "No caption available"
                )

            caption_details.append(
                f"Component {comp_id} (similarity: {similarity:.3f}): {caption}"
            )

        # Create reason with matched component captions
        if len(matched_component_ids) > 0:
            reason = "[CLIP embedding similarity]\n\n" + "\n\n".join(caption_details)
        else:
            reason = f"No matching components found for query: '{query}'"

        return {
            "component_ids": matched_component_ids,
            "reason": reason,
        }
