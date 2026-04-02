import { Form, Button } from "react-bootstrap";
import { useState } from "react";
import { parseResult } from "../SearchResult";

interface BenchmarkInputProps {
  componentIds?: string[];
  onComponentClick?: (index: number) => void;
}

function BenchmarkInput({
  componentIds,
  onComponentClick,
}: BenchmarkInputProps) {
  const [question, setQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [isPreview, setIsPreview] = useState(false);

  const handleSave = () => {
    console.log("Benchmark entry saved (dummy):", { question, expectedAnswer });
  };

  return (
    <div className="d-flex flex-column h-100 pb-2">
      <div className="d-flex flex-column mb-3" style={{ flex: "0 0 auto" }}>
        <div className="d-flex justify-content-between align-items-center mb-1">
          <span className="text-secondary fw-bold">Question</span>
        </div>
        <Form.Control
          as="textarea"
          placeholder="Enter benchmark question here..."
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={{ resize: "none", height: "150px", overflowY: "auto" }}
        />
      </div>

      <div className="d-flex flex-column flex-grow-1 mb-3">
        <div className="d-flex justify-content-between align-items-center mb-1">
          <span className="text-secondary fw-bold">Expected Answer</span>
          <Button
            variant="outline-secondary"
            size="sm"
            onClick={() => setIsPreview(!isPreview)}
          >
            {isPreview ? "Edit" : "Preview"}
          </Button>
        </div>
        {isPreview ? (
          <div
            className="p-2 border rounded bg-light"
            style={{
              flex: 1,
              overflowY: "auto",
              minHeight: "150px",
              whiteSpace: "pre-line",
            }}
          >
            {expectedAnswer ? (
              parseResult(expectedAnswer, componentIds, onComponentClick)
            ) : (
              <span className="text-muted">No content to preview</span>
            )}
          </div>
        ) : (
          <Form.Control
            as="textarea"
            placeholder="Enter expected answer here..."
            value={expectedAnswer}
            onChange={(e) => setExpectedAnswer(e.target.value)}
            style={{
              flex: 1,
              resize: "none",
              minHeight: "150px",
              overflowY: "auto",
            }}
          />
        )}
      </div>

      <div className="mt-auto d-flex justify-content-end">
        <Button variant="primary" onClick={handleSave}>
          Save
        </Button>
      </div>
    </div>
  );
}

export default BenchmarkInput;
