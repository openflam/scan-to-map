import type { SearchResult, SearchQuery } from "./types/global";

const SEARCH_SERVER_URL = "http://172.26.112.246:5000";
// const SEARCH_SERVER_URL = "";

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

    // Return the data as-is from the API
    // Server returns: {"reason": "...", "search_time_ms": ..., "components": [{bbox: {...}, caption: "..."}]}
    return data as SearchResult;
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

export async function getComponentInfo(componentId: string): Promise<{
  component_id: string;
  caption: string;
  image_name: string;
  image_base64: string | null;
  fraction_visible: number;
  image_width: number;
  image_height: number;
}> {
  console.log("Fetching component info for:", componentId);

  try {
    const response = await fetch(
      `${SEARCH_SERVER_URL}/get_component_info?component_id=${encodeURIComponent(componentId)}`,
      {
        method: "GET",
      },
    );

    if (!response.ok) {
      throw new Error(
        `Component info request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Received component info:", data);

    return data;
  } catch (error) {
    console.error("Error fetching component info:", error);
    throw error;
  }
}

export async function deleteComponent(
  componentId: string,
): Promise<{ component_id: string; deleted: boolean }> {
  console.log("Deleting component:", componentId);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/delete_component`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ component_id: componentId }),
    });

    if (!response.ok) {
      throw new Error(
        `Delete component request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Deleted component:", data);

    return data;
  } catch (error) {
    console.error("Error deleting component:", error);
    throw error;
  }
}

export async function updateComponent(
  componentId: string,
  updates: { caption?: string; bbox?: { min: number[]; max: number[] } },
): Promise<{ component_id: string; caption?: string; bbox?: object }> {
  console.log("Updating component:", componentId, updates);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/update_component`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ component_id: componentId, ...updates }),
    });

    if (!response.ok) {
      throw new Error(
        `Update component request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Updated component:", data);

    return data;
  } catch (error) {
    console.error("Error updating component:", error);
    throw error;
  }
}
