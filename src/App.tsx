import { Container, Row } from "react-bootstrap";
import { useState } from "react";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";
import { query } from "./query";
import type { BoundingBox } from "./types/global";

function App() {
  const [boundingBox, setBoundingBox] = useState<BoundingBox | undefined>(
    undefined
  );

  const handleSearch = async (searchTerm: string) => {
    const result = await query(searchTerm);
    setBoundingBox(result);
  };

  return (
    <Container className="pt-3">
      <Row>
        <SearchBar onSearch={handleSearch} />
      </Row>
      <Row style={{ height: "80vh" }}>
        <Model3DViewer
          source={"https://playground.babylonjs.com/scenes/BoomBox.glb"}
          boundingBox={boundingBox}
        />
      </Row>
    </Container>
  );
}

export default App;
