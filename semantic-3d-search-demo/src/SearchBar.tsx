import { InputGroup, FormControl, Button, Form } from "react-bootstrap";
import { useState } from "react";

interface SearchBarProps {
  onSearch: (searchTerm: string) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
}

function SearchBar({ onSearch, showAutoTags, onShowAutoTagsChange }: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState("");

  const handleSearch = () => {
    if (searchTerm.trim()) {
      onSearch(searchTerm);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  return (
    <>
      <InputGroup className="mb-3">
        <Form.Select style={{ maxWidth: "200px" }}>
          <option disabled>Method</option>
          <option value="conceptfusion">ConceptFusion</option>
          <option value="3d-llm">3D-LLM</option>
        </Form.Select>

        <FormControl
          placeholder="Search..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          onKeyDown={handleKeyPress}
        />

        <Button variant="outline-primary" onClick={handleSearch}>
          Search
        </Button>
      </InputGroup>

      <Form.Check
        type="checkbox"
        label="Show Auto Tags"
        checked={showAutoTags}
        onChange={(e) => onShowAutoTagsChange(e.target.checked)}
        className="mb-3"
      />
    </>
  );
}

export default SearchBar;
