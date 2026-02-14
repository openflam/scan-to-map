import type { SearchResult, SearchQuery } from "./types/global";

const SEARCH_SERVER_URL = "http://172.26.112.246:5000";

export async function query(
  searchQuery: SearchQuery,
  method: string,
): Promise<SearchResult> {
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
      throw new Error(
        `Search request failed: ${response.status} ${response.statusText}`,
      );
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

export async function queryDirections(
  source: SearchQuery,
  destination: SearchQuery,
  method: string,
): Promise<{
  path: number[][];
  source_bbox: any;
  destination_bbox: any;
  source_reason: string;
  destination_reason: string;
}> {
  console.log(
    "Querying directions from:",
    source,
    "to:",
    destination,
    "using method:",
    method,
  );

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/get_route`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        source: source,
        destination: destination,
        method: method,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Directions request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();

    return data;
  } catch (error) {
    console.error("Error querying directions:", error);
    throw error;
  }
}
