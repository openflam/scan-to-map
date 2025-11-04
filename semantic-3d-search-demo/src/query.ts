import type { SearchResult } from "./types/global";

const SEARCH_SERVER_URL = "http://172.26.101.175:5000";

export async function query(searchTerm: string, method: string = "gpt-4o-mini [Full]"): Promise<SearchResult> {
  console.log("Querying for:", searchTerm, "with method:", method);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search?query=${encodeURIComponent(searchTerm)}&method=${encodeURIComponent(method)}`);

    if (!response.ok) {
      throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Received response:", data);

    // Extract bounding box and reason from the response
    // Server returns: {"bbox": {...}, "reason": "..."}
    // Wrap single bbox in array for consistency
    const bboxArray = Array.isArray(data.bbox) ? data.bbox : [data.bbox];

    return {
      boundingBox: bboxArray,
      reason: data.reason
    };
  } catch (error) {
    console.error("Error querying search server:", error);
    throw error;
  }
}
