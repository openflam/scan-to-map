# fmt: off

# Prompt for captioning images of indidivudal components
CAPTION_IMAGE_PROMPT = \
"""These images show different views of the same object or region in a 3D scene.
Analyze all the images together and provide a concise, descriptive caption
that captures what this object or region is. Focus on:
1. What the main object/region is
2. Its key visual characteristics (color, shape, texture)
3. Any notable features or context

Keep the caption clear and factual, suitable for 3D semantic search."""

CLASSIFy_IMAGE_PROMPT = \
"""Classify the image as Object or Scene.

Object: A single coherent object is the main focus, even if it has many parts (e.g., a printer, an open computer case). Background should not contain multiple distinct objects.

Scene: The image contains multiple distinct objects or a wide view (room, street, landscape, etc.).

Output only one word: Object or Scene."""
