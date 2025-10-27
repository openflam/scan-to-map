import type { SearchResult } from "./types/global";

const SEARCH_SERVER_URL = "http://localhost:5000";

export async function query(searchTerm: string): Promise<SearchResult> {
  console.log("Querying for:", searchTerm);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search?query=${encodeURIComponent(searchTerm)}`);

    if (!response.ok) {
      throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Received response:", data);

    // Extract bounding box and reason from the response
    // Server returns: {"bbox": {...}, "reason": "..."}
    return {
      boundingBox: data.bbox,
      reason: data.reason
    };
  } catch (error) {
    console.error("Error querying search server:", error);
    throw error;
  }
}
