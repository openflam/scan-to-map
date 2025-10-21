import { InputGroup, FormControl, Button, Form } from "react-bootstrap";

function SearchBar() {
  return (
    <InputGroup className="mb-3">
      <Form.Select style={{ maxWidth: "200px" }}>
        <option disabled>Method</option>
        <option value="conceptfusion">ConceptFusion</option>
        <option value="3d-llm">3D-LLM</option>
      </Form.Select>

      <FormControl placeholder="Search..." />

      <Button variant="outline-primary">Search</Button>
    </InputGroup>
  );
}

export default SearchBar;
