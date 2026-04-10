import { Container } from "react-bootstrap";
import { styles } from "./appStyles";
import { useState, useEffect, useMemo } from "react";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import SearchResult from "./SearchResult";
import SearchComponentList from "./SearchComponentList";
import { query, queryStream, SEARCH_SERVER_URL } from "./query";
import type { BoundingBox, SearchQuery, Route } from "./types/global";

/** Read the dataset name from the `dataset_name` query parameter, e.g. "?dataset_name=ProjectLabNeg" */
function getDatasetNameFromPath(): string | null {
  return new URLSearchParams(window.location.search).get("dataset_name");
}

function App() {
  const datasetName = useMemo(() => getDatasetNameFromPath(), []);

  const [boundingBox, setBoundingBox] = useState<BoundingBox[]>([]);
  const [captions, setCaptions] = useState<string[]>([]);
  const [componentIds, setComponentIds] = useState<string[]>([]);
  const [showAutoTags, setShowAutoTags] = useState(false);
  const [showOccupancyGrid, setShowOccupancyGrid] = useState(false);
  const [autoTagBBoxes, setAutoTagBBoxes] = useState<BoundingBox[]>([]);
  const [occupancyGrid, setOccupancyGrid] = useState<BoundingBox[]>([]);
  const [annotations, setAnnotations] = useState<string[]>([]);
  const [searchResult, setSearchResult] = useState<string | undefined>(
    undefined,
  );
  const [thinking, setThinking] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);
  const [searchTime, setSearchTime] = useState<number | undefined>(undefined);
  const [route, setRoute] = useState<Route>([]);
  const [focusedComponentIndex, setFocusedComponentIndex] = useState<
    number | null
  >(null);
  const [isLeftColumnCollapsed, setIsLeftColumnCollapsed] = useState(false);

  const handleAnnotationsDownloaded = (
    bboxes: BoundingBox[],
    annotationList: string[],
  ) => {
    setAutoTagBBoxes(bboxes);
    setAnnotations(annotationList);
  };

  // Load occupancy_bbox.json on mount
  useEffect(() => {
    fetch("/data/occupancy_bbox.json")
      .then((response) => response.json())
      .then((data) => {
        // Extract bounding boxes from the JSON data.
        // Swap axes as needed -- results of trial and error.
        const bboxes: BoundingBox[] = data.map((item: any) => ({
          corners: item.bbox.corners.map((c: number[]) => [c[1], c[2], c[0]]),
        }));
        setOccupancyGrid(bboxes);
      })
      .catch((error) => {
        console.error("Error loading occupancy_bbox.json:", error);
      });
  }, []);

  const handleSearch = async (searchQuery: SearchQuery, method: string) => {
    setIsLoading(true);
    setSearchResult(undefined);
    setThinking(undefined);
    setFocusedComponentIndex(null);
    setRoute([]); // Clear any existing route

    if (method === "gpt-5.4-tools") {
      let currentThinking = "";
      try {
        await queryStream(searchQuery, method, datasetName!, (event) => {
          if (event.type === "thinking") {
            currentThinking += event.content;
            setThinking(currentThinking);
          } else if (event.type === "result") {
            const result = event.data;
            setBoundingBox(result.components.map((c: any) => c.bbox));
            setCaptions(result.components.map((c: any) => c.caption));
            setComponentIds(result.components.map((c: any) => c.component_id));
            setSearchResult(result.reason);
            setSearchTime(result.search_time_ms);
          } else if (event.type === "error") {
            console.error("Stream error:", event.error);
            setSearchResult(`Error: ${event.error}`);
          }
        });
      } catch (e) {
        console.error("Query stream failed:", e);
        setSearchResult("Search failed due to a network or server error.");
      } finally {
        setIsLoading(false);
      }
    } else {
      const result = await query(searchQuery, method, datasetName!);
      // Extract bounding boxes, captions, and component IDs from components
      setBoundingBox(result.components.map((c) => c.bbox));
      setCaptions(result.components.map((c) => c.caption));
      setComponentIds(result.components.map((c) => c.component_id));
      setSearchResult(result.reason);
      setSearchTime(result.search_time_ms);
      setIsLoading(false);
    }
  };

  const handleDirections = (
    route: Route,
    sourceBBox: BoundingBox,
    destinationBBox: BoundingBox,
    sourceReason: string,
    destinationReason: string,
  ) => {
    setRoute(route);
    // Set the bounding boxes to display source and destination
    setBoundingBox([sourceBBox, destinationBBox]);
    setCaptions(["Source", "Destination"]);
    setComponentIds([]); // Clear component IDs for directions mode
    setFocusedComponentIndex(null);
    // Combine the reasons
    const combinedReason = `Source: ${sourceReason}\n\nDestination: ${destinationReason}`;
    setSearchResult(combinedReason);
  };

  if (!datasetName) {
    return (
      <Container className="pt-5 text-center">
        <h1>404 — Dataset Not Found</h1>
        <p className="text-muted">No dataset name was provided in the URL.</p>
      </Container>
    );
  }

  return (
    <div style={styles.rootContainer}>
      {/* Top Search Bar Area */}
      <div style={styles.topBarArea}>
        <SearchBar
          onSearch={handleSearch}
          onDirections={handleDirections}
          onAnnotationsDownloaded={handleAnnotationsDownloaded}
          showAutoTags={showAutoTags}
          onShowAutoTagsChange={setShowAutoTags}
          showOccupancyGrid={showOccupancyGrid}
          onShowOccupancyGridChange={setShowOccupancyGrid}
          searchTime={searchTime}
          datasetName={datasetName}
        />
      </div>

      {/* Main Content Area */}
      <div style={styles.mainContentArea}>
        {/* Collapsible Left Column */}
        <div
          style={{
            ...styles.leftColumnBase,
            width: isLeftColumnCollapsed ? "0px" : "460px",
            borderRight: isLeftColumnCollapsed ? "none" : "1px solid #dee2e6",
          }}
        >
          {/* Search Result Box */}
          <div style={styles.searchResultBox}>
            <SearchResult
              result={searchResult}
              thinking={thinking}
              isLoading={isLoading}
              componentIds={componentIds}
              onComponentClick={(i: number) => setFocusedComponentIndex(i)}
            />
          </div>

          {/* Search Component List */}
          <div style={styles.searchComponentListWrapper}>
            <SearchComponentList
              componentIds={componentIds}
              captions={captions}
              datasetName={datasetName}
              onComponentClick={(i) => setFocusedComponentIndex(i)}
              focusedComponentIndex={focusedComponentIndex}
            />
          </div>
        </div>

        {/* Collapse Toggle Button */}
        <button
          onClick={() => setIsLeftColumnCollapsed(!isLeftColumnCollapsed)}
          style={styles.collapseToggleButton}
          title={isLeftColumnCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isLeftColumnCollapsed ? "▶" : "◀"}
        </button>

        {/* 3D Viewer */}
        <div style={styles.viewerContainer}>
          <Model3DViewer
            source={`${SEARCH_SERVER_URL}/load_mesh?dataset_name=${encodeURIComponent(datasetName)}`}
            boundingBox={boundingBox}
            captions={captions}
            componentIds={componentIds}
            autoTagBBoxes={autoTagBBoxes}
            showAutoTags={showAutoTags}
            occupancyGrid={occupancyGrid}
            showOccupancyGrid={showOccupancyGrid}
            annotations={annotations}
            route={route}
            datasetName={datasetName}
            focusedBBoxIndex={focusedComponentIndex}
            externalSelectedBBoxIndex={focusedComponentIndex}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
