import { InputGroup, FormControl, Button, Form } from "react-bootstrap";
import { useState } from "react";

interface SearchBarProps {
  onSearch: (searchTerm: string, method: string) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
  searchTime?: number;
}

function SearchBar({ onSearch, showAutoTags, onShowAutoTagsChange, searchTime }: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [method, setMethod] = useState("gpt-4o-mini [Full]");

  const handleSearch = () => {
    if (searchTerm.trim()) {
      onSearch(searchTerm, method);
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
        <Form.Select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          style={{ maxWidth: "200px" }}
        >
          <option value="gpt-4o-mini [Full]">gpt-4o-mini [Full]</option>
          <option value="gpt-4o-mini [RAG]">gpt-4o-mini [RAG]</option>
          <option value="BM25">BM25</option>
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

      <div className="d-flex align-items-center justify-content-between mb-3">
        <Form.Check
          type="checkbox"
          label="Show Auto Tags"
          checked={showAutoTags}
          onChange={(e) => onShowAutoTagsChange(e.target.checked)}
        />
        <div className="d-flex align-items-center" style={{ fontSize: "1.1rem", fontWeight: "500" }}>
          <i className="bi bi-clock me-2"></i>
          {searchTime !== undefined ? `${searchTime.toFixed(2)} ms` : '...'}
        </div>
      </div>
    </>
  );
}

export default SearchBar;
