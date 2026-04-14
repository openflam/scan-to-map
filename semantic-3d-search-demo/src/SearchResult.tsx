import { Spinner, Accordion, Form } from "react-bootstrap";
import React, { useState } from "react";

interface SearchResultProps {
  result?: string;
  thinking?: string;
  isLoading?: boolean;
  componentIds?: string[];
  onComponentClick?: (index: number) => void;
}

export const parseResult = (
  text: string,
  componentIds?: string[],
  onComponentClick?: (index: number) => void
) => {
  const regex = /<component_(\d+)>(.*?)<\/component_\1>/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }

    const componentId = match[1];
    const componentName = match[2];

    const index = componentIds?.findIndex((id) => String(id) === componentId);

    if (index !== undefined && index !== -1) {
      parts.push(
        <a
          key={match.index}
          href="#"
          className="text-decoration-none fw-bold"
          onClick={(e) => {
            e.preventDefault();
            if (onComponentClick) onComponentClick(index);
          }}
        >
          {componentName}
        </a>,
      );
    } else {
      parts.push(
        <span
          key={match.index}
          className="fw-bold text-decoration-underline"
          style={{ cursor: "pointer" }}
          onClick={() =>
            console.warn(
              "Component ID not found in results list",
              componentId,
            )
          }
        >
          {componentName}
        </span>,
      );
    }

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  return parts;
};

function SearchResult({
  result,
  thinking,
  isLoading,
  componentIds,
  onComponentClick,
}: SearchResultProps) {
  const [isRaw, setIsRaw] = useState(false);

  return (
    <>
      {thinking && (
        <Accordion className="mb-3" defaultActiveKey="0">
          <Accordion.Item eventKey="0">
            <Accordion.Header>
              <div className="d-flex align-items-center">
                {isLoading && (
                  <Spinner
                    animation="grow"
                    size="sm"
                    variant="secondary"
                    className="me-2"
                  />
                )}
                <span className="text-secondary">
                  {isLoading ? "Thinking..." : "Thought Process"}
                </span>
              </div>
            </Accordion.Header>
            <Accordion.Body
              style={{
                whiteSpace: "pre-line",
                maxHeight: "300px",
                overflowY: "auto",
              }}
            >
              {thinking}
            </Accordion.Body>
          </Accordion.Item>
        </Accordion>
      )}
      {result && (
        <div className="mb-2">
          <Form.Check
            type="switch"
            id="raw-mode-switch"
            label={<span className="text-muted small">Raw mode</span>}
            checked={isRaw}
            onChange={(e) => setIsRaw(e.target.checked)}
          />
        </div>
      )}
      <div className="d-flex justify-content-between align-items-start">
        <div className="flex-grow-1" style={{ minWidth: 0 }}>
          {isLoading && !result && !thinking ? (
            <div className="d-flex align-items-center gap-2 text-muted">
              <Spinner animation="border" size="sm" role="status" />
              <span>Loading...</span>
            </div>
          ) : result ? (
            <p className="mb-0 text-break" style={{ whiteSpace: "pre-line", wordBreak: "break-word" }}>
              {isRaw ? result : parseResult(result, componentIds, onComponentClick)}
            </p>
          ) : !thinking ? (
            <p className="text-muted mb-0">No search results yet</p>
          ) : null}
        </div>
        {(result || (isLoading && !thinking)) && (
          <span className="badge bg-secondary ms-3">Result</span>
        )}
      </div>
    </>
  );
}

export default SearchResult;
