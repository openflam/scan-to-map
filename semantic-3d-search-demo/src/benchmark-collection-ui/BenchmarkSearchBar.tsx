import {
  InputGroup,
  FormControl,
  Button,
  Form,
  ProgressBar,
} from "react-bootstrap";
import { useState } from "react";
import type { BoundingBox, SearchQuery, Route } from "../types/global";
import { queryDirections, downloadAllComponents } from "../query";

interface SearchBarProps {
  onSearch: (searchQuery: SearchQuery, method: string) => void;
  onDirections: (
    route: Route,
    sourceBBox: any,
    destinationBBox: any,
    sourceReason: string,
    destinationReason: string,
  ) => void;
  onAnnotationsDownloaded: (
    bboxes: BoundingBox[],
    annotations: string[],
  ) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
  showOccupancyGrid: boolean;
  onShowOccupancyGridChange: (show: boolean) => void;
  searchTime?: number;
  datasetName: string;
}

function BenchmarkSearchBar({
  onSearch,
  onDirections,
  onAnnotationsDownloaded,
  showAutoTags,
  onShowAutoTagsChange,
  showOccupancyGrid,
  onShowOccupancyGridChange,
  searchTime,
  datasetName,
}: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const method = "BM25"; // Fixed to BM25
  const [showDirections, setShowDirections] = useState(false);
  const [sourceTerm, setSourceTerm] = useState("");
  const [destinationTerm, setDestinationTerm] = useState("");
  const [isDownloadingAnnotations, setIsDownloadingAnnotations] =
    useState(false);
  const [annotationsLoaded, setAnnotationsLoaded] = useState(false);

  const handleDownloadAnnotations = async () => {
    setIsDownloadingAnnotations(true);
    try {
      const data = await downloadAllComponents(datasetName);
      const bboxes: BoundingBox[] = data.map((item: any) => ({
        corners: item.bbox.corners.map((c: number[]) => [c[1], c[2], c[0]]),
      }));
      const annotationList: string[] = data.map((item) =>
        item.connected_comp_id.toString(),
      );
      onAnnotationsDownloaded(bboxes, annotationList);
      setAnnotationsLoaded(true);
    } catch (error) {
      console.error("Failed to download annotations:", error);
    } finally {
      setIsDownloadingAnnotations(false);
    }
  };

  const handleSearch = () => {
    if (showDirections) {
      handleDirections();
      return;
    }

    if (searchTerm.trim()) {
      const searchQuery: SearchQuery = [];
      searchQuery.push({ type: "text", value: searchTerm });
      onSearch(searchQuery, method);
    }
  };

  const handleDirections = async () => {
    if (!sourceTerm.trim() || !destinationTerm.trim()) {
      console.error("Both source and destination are required");
      return;
    }

    try {
      const sourceQuery: SearchQuery = [{ type: "text", value: sourceTerm }];
      const destinationQuery: SearchQuery = [
        { type: "text", value: destinationTerm },
      ];

      const result = await queryDirections(
        sourceQuery,
        destinationQuery,
        method,
        datasetName,
      );
      console.log("Route received:", result);
      const route: Route = result.path.map(
        (coord) => [coord[0], coord[1], coord[2]] as [number, number, number],
      );
      onDirections(
        route,
        result.source_bbox,
        result.destination_bbox,
        result.source_reason,
        result.destination_reason,
      );
    } catch (error) {
      console.error("Failed to get directions:", error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
  };

  return (
    <>
      <InputGroup className="mb-3">
        <Form.Select value={method} disabled style={{ maxWidth: "200px" }}>
          <option value="BM25">BM25</option>
        </Form.Select>

        {!showDirections ? (
          <>
            <FormControl
              placeholder="Search..."
              value={searchTerm}
              onChange={handleTextChange}
              onKeyDown={handleKeyPress}
            />

            <Button variant="outline-primary" onClick={handleSearch}>
              Search
            </Button>
          </>
        ) : (
          <>
            <FormControl
              placeholder="Source..."
              value={sourceTerm}
              onChange={(e) => setSourceTerm(e.target.value)}
              onKeyDown={handleKeyPress}
            />
            <FormControl
              placeholder="Destination..."
              value={destinationTerm}
              onChange={(e) => setDestinationTerm(e.target.value)}
              onKeyDown={handleKeyPress}
            />
            <Button variant="outline-primary" onClick={handleSearch}>
              Get Directions
            </Button>
          </>
        )}
      </InputGroup>

      <div className="d-flex align-items-center justify-content-between mb-3">
        <div className="d-flex align-items-center gap-3">
          {isDownloadingAnnotations ? (
            <ProgressBar
              animated
              striped
              now={100}
              label="Downloading…"
              style={{ width: "160px", height: "31px" }}
            />
          ) : (
            <Button
              variant={annotationsLoaded ? "secondary" : "outline-secondary"}
              disabled={annotationsLoaded}
              onClick={handleDownloadAnnotations}
            >
              {annotationsLoaded
                ? "Annotations Downloaded"
                : "Download Annotations"}
            </Button>
          )}
          <Form.Check
            type="checkbox"
            label="Show Auto Tags"
            checked={showAutoTags}
            disabled={!annotationsLoaded}
            onChange={(e) => onShowAutoTagsChange(e.target.checked)}
          />
          <Form.Check
            type="checkbox"
            label="Show Occupancy Grid"
            checked={showOccupancyGrid}
            onChange={(e) => onShowOccupancyGridChange(e.target.checked)}
          />
          <Form.Check
            type="checkbox"
            label="Directions"
            checked={showDirections}
            onChange={(e) => setShowDirections(e.target.checked)}
          />
        </div>
        <div
          className="d-flex align-items-center"
          style={{ fontSize: "1.1rem", fontWeight: "500" }}
        >
          <i className="bi bi-clock me-2"></i>
          {searchTime !== undefined ? `${searchTime.toFixed(2)} ms` : "..."}
        </div>
      </div>
    </>
  );
}

export default BenchmarkSearchBar;
