import { Container, Row } from "react-bootstrap";
import { useState, useEffect, useMemo } from "react";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import SearchResult from "./SearchResult";
import SearchComponentList from "./SearchComponentList";
import { query, SEARCH_SERVER_URL } from "./query";
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
  const [isLoading, setIsLoading] = useState(false);
  const [searchTime, setSearchTime] = useState<number | undefined>(undefined);
  const [route, setRoute] = useState<Route>([]);
  const [focusedComponentIndex, setFocusedComponentIndex] = useState<
    number | null
  >(null);

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
          x_min: item.bbox.min[1],
          y_min: item.bbox.min[2],
          z_min: item.bbox.min[0],
          x_max: item.bbox.max[1],
          y_max: item.bbox.max[2],
          z_max: item.bbox.max[0],
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
    setFocusedComponentIndex(null);
    const result = await query(searchQuery, method, datasetName!);
    setRoute([]); // Clear any existing route
    // Extract bounding boxes, captions, and component IDs from components
    setBoundingBox(result.components.map((c) => c.bbox));
    setCaptions(result.components.map((c) => c.caption));
    setComponentIds(result.components.map((c) => c.component_id));
    setSearchResult(result.reason);
    setSearchTime(result.search_time_ms);
    setIsLoading(false);
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
    <Container className="pt-3">
      <Row>
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
      </Row>
      <Row style={{ height: "80vh" }}>
        <div style={{ display: "flex", height: "100%", padding: 0 }}>
          <SearchComponentList
            componentIds={componentIds}
            captions={captions}
            datasetName={datasetName}
            onComponentClick={(i) => setFocusedComponentIndex(i)}
            focusedComponentIndex={focusedComponentIndex}
          />
          <div style={{ flex: 1, minWidth: 0, height: "100%" }}>
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
      </Row>
      <Row>
        <SearchResult result={searchResult} isLoading={isLoading} />
      </Row>
    </Container>
  );
}

export default App;
