import type { BoundingBox } from "./types/global";

// TEMPORARY: Mock query function that returns a random bounding box
export async function query(searchTerm: string): Promise<BoundingBox> {
  console.log("Querying for:", searchTerm);

  // Base bounding box coordinates
  const baseBox = {
    x_min: -0.009921154,
    y_min: -0.00977163,
    z_min: -0.0100762453,
    x_max: 0.009921154,
    y_max: 0.00977163,
    z_max: 0.0100762453,
  };

  // Generate random fractions for each axis (between 0.3 and 1.0 to keep boxes visible)
  const xFraction = 0.3 + Math.random() * 0.7;
  const yFraction = 0.3 + Math.random() * 0.7;
  const zFraction = 0.3 + Math.random() * 0.7;

  // Calculate the center of the base box
  const xCenter = (baseBox.x_min + baseBox.x_max) / 2;
  const yCenter = (baseBox.y_min + baseBox.y_max) / 2;
  const zCenter = (baseBox.z_min + baseBox.z_max) / 2;

  // Calculate half-widths of the base box
  const xHalfWidth = (baseBox.x_max - baseBox.x_min) / 2;
  const yHalfWidth = (baseBox.y_max - baseBox.y_min) / 2;
  const zHalfWidth = (baseBox.z_max - baseBox.z_min) / 2;

  // Return a fraction of the bounding box, centered at the same point
  return {
    x_min: xCenter - xHalfWidth * xFraction,
    y_min: yCenter - yHalfWidth * yFraction,
    z_min: zCenter - zHalfWidth * zFraction,
    x_max: xCenter + xHalfWidth * xFraction,
    y_max: yCenter + yHalfWidth * yFraction,
    z_max: zCenter + zHalfWidth * zFraction,
  };
}
