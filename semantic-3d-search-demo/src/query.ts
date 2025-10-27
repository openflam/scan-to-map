import type { BoundingBox } from "./types/global";

const SEARCH_SERVER_URL = "http://localhost:5000";

export async function query(searchTerm: string): Promise<BoundingBox> {
  console.log("Querying for:", searchTerm);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search?query=${encodeURIComponent(searchTerm)}`);

    if (!response.ok) {
      throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
    }

    const boundingBox: BoundingBox = await response.json();
    console.log("Received bounding box:", boundingBox);

    return boundingBox;
  } catch (error) {
    console.error("Error querying search server:", error);
    throw error;
  }
}
