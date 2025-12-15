import { Container, Row } from "react-bootstrap";
import { useState, useEffect } from "react";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import SearchResult from "./SearchResult";
import { query } from "./query";
import type { BoundingBox, SearchQuery, Route } from "./types/global";

function App() {
  const [boundingBox, setBoundingBox] = useState<BoundingBox[]>([]);
  const [showAutoTags, setShowAutoTags] = useState(false);
  const [showOccupancyGrid, setShowOccupancyGrid] = useState(false);
  const [autoTagBBoxes, setAutoTagBBoxes] = useState<BoundingBox[]>([]);
  const [occupancyGrid, setOccupancyGrid] = useState<BoundingBox[]>([]);
  const [annotations, setAnnotations] = useState<string[]>([]);
  const [searchResult, setSearchResult] = useState<string | undefined>(
    undefined
  );
  const [searchTime, setSearchTime] = useState<number | undefined>(undefined);
  const [route, setRoute] = useState<Route>([]);

  // Load bbox_corners.json on mount
  useEffect(() => {
    fetch("/data/bbox_corners.json")
      .then((response) => response.json())
      .then((data) => {
        // Extract bounding boxes from the JSON data.
        // Swap axes as needed -- results of trial and error.
        const bboxes: BoundingBox[] = data.map((item: any) => ({
          x_min: -item.bbox.min[1],
          y_min: item.bbox.min[2],
          z_min: item.bbox.min[0],
          x_max: -item.bbox.max[1],
          y_max: item.bbox.max[2],
          z_max: item.bbox.max[0],
        }));
        setAutoTagBBoxes(bboxes);

        // Extract annotations (connected_comp_id)
        const annotationList: string[] = data.map((item: any) =>
          item.connected_comp_id.toString()
        );
        setAnnotations(annotationList);
      })
      .catch((error) => {
        console.error("Error loading bbox_corners.json:", error);
      });
  }, []);

  // Load occupancy_bbox.json on mount
  useEffect(() => {
    fetch("/data/occupancy_bbox.json")
      .then((response) => response.json())
      .then((data) => {
        // Extract bounding boxes from the JSON data.
        // Swap axes as needed -- results of trial and error.
        const bboxes: BoundingBox[] = data.map((item: any) => ({
          x_min: -item.bbox.min[1],
          y_min: item.bbox.min[2],
          z_min: item.bbox.min[0],
          x_max: -item.bbox.max[1],
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
    const result = await query(searchQuery, method);
    setRoute([]); // Clear any existing route
    setBoundingBox(result.boundingBox);
    setSearchResult(result.reason);
    setSearchTime(result.searchTimeMs);
  };

  const handleDirections = (
    route: Route,
    sourceBBox: BoundingBox,
    destinationBBox: BoundingBox,
    sourceReason: string,
    destinationReason: string
  ) => {
    setRoute(route);
    // Set the bounding boxes to display source and destination
    setBoundingBox([sourceBBox, destinationBBox]);
    // Combine the reasons
    const combinedReason = `Source: ${sourceReason}\n\nDestination: ${destinationReason}`;
    setSearchResult(combinedReason);
  };

  return (
    <Container className="pt-3">
      <Row>
        <SearchBar
          onSearch={handleSearch}
          onDirections={handleDirections}
          showAutoTags={showAutoTags}
          onShowAutoTagsChange={setShowAutoTags}
          showOccupancyGrid={showOccupancyGrid}
          onShowOccupancyGridChange={setShowOccupancyGrid}
          searchTime={searchTime}
        />
      </Row>
      <Row style={{ height: "80vh" }}>
        <Model3DViewer
          source={"/data/raw.glb"}
          boundingBox={boundingBox}
          autoTagBBoxes={autoTagBBoxes}
          showAutoTags={showAutoTags}
          occupancyGrid={occupancyGrid}
          showOccupancyGrid={showOccupancyGrid}
          annotations={annotations}
          route={route}
        />
      </Row>
      <Row>
        <SearchResult result={searchResult} />
      </Row>
    </Container>
  );
}

export default App;
