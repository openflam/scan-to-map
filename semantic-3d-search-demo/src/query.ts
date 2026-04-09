import type { SearchResult, SearchQuery } from "./types/global";

// export const SEARCH_SERVER_URL = "http://172.26.112.246:5000";
export const SEARCH_SERVER_URL = "";

export async function getProvidersList(): Promise<string[]> {
  const response = await fetch(`${SEARCH_SERVER_URL}/get_providers_list`, {
    method: "GET",
  });
  if (!response.ok) {
    throw new Error(
      `Get providers list failed: ${response.status} ${response.statusText}`,
    );
  }
  const data = await response.json();
  return data.providers as string[];
}

export async function query(
  searchQuery: SearchQuery,
  method: string,
  datasetName: string,
): Promise<SearchResult> {
  console.log("Querying with:", searchQuery, "using method:", method);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
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

export async function queryStream(
  searchQuery: SearchQuery,
  method: string,
  datasetName: string,
  onEvent: (eventData: any) => void
): Promise<void> {
  console.log("Streaming query with:", searchQuery, "using method:", method);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/search_stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        query: searchQuery,
        method: method,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(
        `Search stream request failed: ${response.status} ${response.statusText}`
      );
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");

      buffer = lines.pop() || ""; // Keep the last incomplete line in the buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6);
          try {
            const data = JSON.parse(dataStr);
            onEvent(data);
          } catch (e) {
            console.error("Error parsing stream data:", e, dataStr);
          }
        }
      }
    }
  } catch (error) {
    console.error("Error streaming search server:", error);
    throw error;
  }
}

export async function queryDirections(
  source: SearchQuery,
  destination: SearchQuery,
  method: string,
  datasetName: string,
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
        dataset_name: datasetName,
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

export async function getComponentInfo(
  componentId: string,
  datasetName: string,
): Promise<{
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
      `${SEARCH_SERVER_URL}/get_component_info?dataset_name=${encodeURIComponent(datasetName)}&component_id=${encodeURIComponent(componentId)}`,
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
  datasetName: string,
): Promise<{ component_id: string; deleted: boolean }> {
  console.log("Deleting component:", componentId);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/delete_component`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        component_id: componentId,
      }),
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

export async function downloadAllComponents(
  datasetName: string,
): Promise<
  Array<{ connected_comp_id: number; bbox: { corners: [number, number, number][] } }>
> {
  console.log("Downloading all components...");

  try {
    const response = await fetch(
      `${SEARCH_SERVER_URL}/download_all_components?dataset_name=${encodeURIComponent(datasetName)}`,
      { method: "GET" },
    );

    if (!response.ok) {
      throw new Error(
        `Download failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log(`Downloaded ${data.length} components`);
    return data;
  } catch (error) {
    console.error("Error downloading all components:", error);
    throw error;
  }
}

export async function updateComponent(
  componentId: string,
  updates: { caption?: string; bbox?: { corners: [number, number, number][] } },
  datasetName: string,
): Promise<{ component_id: string; caption?: string; bbox?: object }> {
  console.log("Updating component:", componentId, updates);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/update_component`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        component_id: componentId,
        ...updates,
      }),
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

export async function callTool(
  toolName: string,
  args: any,
  datasetName: string,
): Promise<any> {
  console.log("Calling tool:", toolName, args);

  try {
    const response = await fetch(`${SEARCH_SERVER_URL}/call_tool`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        dataset_name: datasetName,
        tool_name: toolName,
        arguments: args,
      }),
    });

    if (!response.ok) {
      throw new Error(
        `Call tool request failed: ${response.status} ${response.statusText}`,
      );
    }

    const data = await response.json();
    console.log("Tool result:", data);

    return data;
  } catch (error) {
    console.error("Error calling tool:", error);
    throw error;
  }
}
