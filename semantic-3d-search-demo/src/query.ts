import type { SearchResult, SearchQuery } from "./types/global";

const SEARCH_SERVER_URL = "http://172.26.101.175:5000";

export async function query(searchQuery: SearchQuery, method: string = "gpt-4o-mini [Full]"): Promise<SearchResult> {
  console.log("Querying with:", searchQuery, "using method:", method);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        query: searchQuery,
        method: method,
      }),
    });

    if (!response.ok) {
      throw new Error(`Search request failed: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();
    console.log("Received response:", data);

    // Extract bounding box, reason, and search time from the response
    // Server returns: {"bbox": {...}, "reason": "...", "search_time_ms": ...}
    // Wrap single bbox in array for consistency
    const bboxArray = Array.isArray(data.bbox) ? data.bbox : [data.bbox];

    return {
      boundingBox: bboxArray,
      reason: data.reason,
      searchTimeMs: data.search_time_ms,
    };
  } catch (error) {
    console.error("Error querying search server:", error);
    throw error;
  }
}
