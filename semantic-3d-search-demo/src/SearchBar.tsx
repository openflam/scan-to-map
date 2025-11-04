import { InputGroup, FormControl, Button, Form } from "react-bootstrap";
import { useState } from "react";

interface SearchBarProps {
  onSearch: (searchTerm: string, method: string) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
}

function SearchBar({ onSearch, showAutoTags, onShowAutoTagsChange }: SearchBarProps) {
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
