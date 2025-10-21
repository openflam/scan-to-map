import { Container, Row } from "react-bootstrap";

import SearchBar from "./SearchBar";
import Model3DViewer from "./Model3DViewer";

function App() {
  return (
    <Container className="pt-3">
      <Row>
        <SearchBar />
      </Row>
      <Row style={{ height: "80vh" }}>
        <Model3DViewer
          source={"https://playground.babylonjs.com/scenes/BoomBox.glb"}
        />
      </Row>
    </Container>
  );
}

export default App;
