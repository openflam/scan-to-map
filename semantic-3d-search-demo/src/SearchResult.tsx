import { Card, Spinner } from "react-bootstrap";

interface SearchResultProps {
  result?: string;
  isLoading?: boolean;
}

function SearchResult({ result, isLoading }: SearchResultProps) {
  return (
    <Card className="mt-3">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-start">
          <div className="flex-grow-1">
            {isLoading ? (
              <div className="d-flex align-items-center gap-2 text-muted">
                <Spinner animation="border" size="sm" role="status" />
                <span>Loading...</span>
              </div>
            ) : result ? (
              <p className="mb-0" style={{ whiteSpace: "pre-line" }}>
                {result}
              </p>
            ) : (
              <p className="text-muted mb-0">No search results yet</p>
            )}
          </div>
          <span className="badge bg-secondary ms-3">Result</span>
        </div>
      </Card.Body>
    </Card>
  );
}

export default SearchResult;
