import { Card, Spinner, Accordion } from "react-bootstrap";

interface SearchResultProps {
  result?: string;
  thinking?: string;
  isLoading?: boolean;
}

function SearchResult({ result, thinking, isLoading }: SearchResultProps) {
  return (
    <Card className="mt-3">
      <Card.Body>
        {thinking && (
          <Accordion className="mb-3" defaultActiveKey="0">
            <Accordion.Item eventKey="0">
              <Accordion.Header>
                <div className="d-flex align-items-center">
                  {isLoading && <Spinner animation="grow" size="sm" variant="secondary" className="me-2" />}
                  <span className="text-secondary">{isLoading ? "Thinking..." : "Thought Process"}</span>
                </div>
              </Accordion.Header>
              <Accordion.Body style={{ whiteSpace: "pre-line", maxHeight: "300px", overflowY: "auto" }}>
                {thinking}
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        )}
        <div className="d-flex justify-content-between align-items-start">
          <div className="flex-grow-1">
            {isLoading && !result && !thinking ? (
              <div className="d-flex align-items-center gap-2 text-muted">
                <Spinner animation="border" size="sm" role="status" />
                <span>Loading...</span>
              </div>
            ) : result ? (
              <p className="mb-0" style={{ whiteSpace: "pre-line" }}>
                {result}
              </p>
            ) : !thinking ? (
              <p className="text-muted mb-0">No search results yet</p>
            ) : null}
          </div>
          {(result || (isLoading && !thinking)) && <span className="badge bg-secondary ms-3">Result</span>}
        </div>
      </Card.Body>
    </Card>
  );
}

export default SearchResult;
