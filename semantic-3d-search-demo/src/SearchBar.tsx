import { InputGroup, FormControl, Button, Form } from "react-bootstrap";
import { useState, useRef } from "react";
import type { SearchQuery } from "./types/global";

interface SearchBarProps {
  onSearch: (searchQuery: SearchQuery, method: string) => void;
  showAutoTags: boolean;
  onShowAutoTagsChange: (show: boolean) => void;
  showOccupancyGrid: boolean;
  onShowOccupancyGridChange: (show: boolean) => void;
  searchTime?: number;
}

function SearchBar({
  onSearch,
  showAutoTags,
  onShowAutoTagsChange,
  showOccupancyGrid,
  onShowOccupancyGridChange,
  searchTime,
}: SearchBarProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [method, setMethod] = useState("CLIP ViT-H-14");
  const [uploadedImage, setUploadedImage] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSearch = () => {
    if (searchTerm.trim() || uploadedImage) {
      const searchQuery: SearchQuery = [];

      if (searchTerm.trim()) {
        searchQuery.push({ type: "text", value: searchTerm });
      }

      if (uploadedImage) {
        // Convert image to base64 for transmission
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result as string;
          searchQuery.push({ type: "image", value: base64String });
          onSearch(searchQuery, method);
        };
        reader.readAsDataURL(uploadedImage);
        return;
      }

      onSearch(searchQuery, method);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSearch();
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedImage(file);
      setSearchTerm(""); // Clear text when image is uploaded
    }
  };

  const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchTerm(e.target.value);
    if (e.target.value.trim() && uploadedImage) {
      setUploadedImage(null); // Clear image when text is entered
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleClearImage = () => {
    setUploadedImage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const isClipMethod = method === "CLIP ViT-H-14";

  return (
    <>
      <InputGroup className="mb-3">
        <Form.Select
          value={method}
          onChange={(e) => setMethod(e.target.value)}
          style={{ maxWidth: "200px" }}
        >
          <option value="CLIP ViT-H-14">CLIP ViT-H-14</option>
          <option value="gpt-4o-mini [Full]">gpt-4o-mini [Full]</option>
          <option value="gpt-5-mini [RAG]">gpt-5-mini [RAG]</option>
          <option value="BM25">BM25</option>
        </Form.Select>

        <FormControl
          placeholder="Search..."
          value={searchTerm}
          onChange={handleTextChange}
          onKeyDown={handleKeyPress}
          disabled={!!uploadedImage}
        />

        {isClipMethod && (
          <>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageUpload}
              accept="image/*"
              style={{ display: "none" }}
            />
            <Button
              variant={uploadedImage ? "success" : "outline-primary"}
              onClick={() => fileInputRef.current?.click()}
              disabled={!!searchTerm.trim()}
              title="Upload image"
            >
              <i className="bi bi-image"></i>
              {!uploadedImage && " Upload Image"}
              {uploadedImage && ` ${uploadedImage.name}`}
            </Button>
            {uploadedImage && (
              <Button
                variant="outline-danger"
                onClick={handleClearImage}
                title="Clear image"
              >
                <i className="bi bi-x-circle"></i>
              </Button>
            )}
          </>
        )}

        <Button variant="outline-primary" onClick={handleSearch}>
          Search
        </Button>
      </InputGroup>

      <div className="d-flex align-items-center justify-content-between mb-3">
        <div>
          <Form.Check
            type="checkbox"
            label="Show Auto Tags"
            checked={showAutoTags}
            onChange={(e) => onShowAutoTagsChange(e.target.checked)}
          />
          <Form.Check
            type="checkbox"
            label="Show Occupancy Grid"
            checked={showOccupancyGrid}
            onChange={(e) => onShowOccupancyGridChange(e.target.checked)}
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

export default SearchBar;
