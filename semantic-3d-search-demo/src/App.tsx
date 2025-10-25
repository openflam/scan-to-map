import { Container, Row } from "react-bootstrap";
import { useState, useEffect } from "react";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import { query } from "./query";
import type { BoundingBox } from "./types/global";

function App() {
  const [boundingBox, setBoundingBox] = useState<BoundingBox | undefined>(
    undefined
  );
  const [showAutoTags, setShowAutoTags] = useState(false);
  const [autoTagBBoxes, setAutoTagBBoxes] = useState<BoundingBox[]>([]);

  // Load bbox_corners.json on mount
  useEffect(() => {
    fetch("/data/bbox_corners.json")
      .then((response) => response.json())
      .then((data) => {
        // Extract bounding boxes from the JSON data
        const bboxes: BoundingBox[] = data.map((item: any) => ({
          x_min: item.bbox.min[0],
          y_min: item.bbox.min[1],
          z_min: item.bbox.min[2],
          x_max: item.bbox.max[0],
          y_max: item.bbox.max[1],
          z_max: item.bbox.max[2],
        }));
        setAutoTagBBoxes(bboxes);
      })
      .catch((error) => {
        console.error("Error loading bbox_corners.json:", error);
      });
  }, []);

  const handleSearch = async (searchTerm: string) => {
    const result = await query(searchTerm);
    setBoundingBox(result);
  };

  return (
    <Container className="pt-3">
      <Row>
        <SearchBar
          onSearch={handleSearch}
          showAutoTags={showAutoTags}
          onShowAutoTagsChange={setShowAutoTags}
        />
      </Row>
      <Row style={{ height: "80vh" }}>
        <Model3DViewer
          source={"/data/raw.glb"}
          boundingBox={boundingBox}
          autoTagBBoxes={autoTagBBoxes}
          showAutoTags={showAutoTags}
        />
      </Row>
    </Container>
  );
}

export default App;
