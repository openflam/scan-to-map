import { Card } from "react-bootstrap";

interface SearchResultProps {
    result?: string;
}

function SearchResult({ result }: SearchResultProps) {
    return (
        <Card className="mt-3">
            <Card.Body>
                <div className="d-flex justify-content-between align-items-start">
                    <div className="flex-grow-1">
                        {result ? (
                            <p className="mb-0">{result}</p>
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
