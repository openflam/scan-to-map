import { Form } from "react-bootstrap";
import { useState } from "react";

function BenchmarkResult() {
  const [expectedAnswer, setExpectedAnswer] = useState("");

  return (
    <div className="d-flex flex-column h-100">
      <div className="d-flex justify-content-between align-items-center mb-2">
        <span className="text-secondary fw-bold">Expected Answer</span>
      </div>
      <Form.Control
        as="textarea"
        placeholder="Enter expected answer here..."
        value={expectedAnswer}
        onChange={(e) => setExpectedAnswer(e.target.value)}
        style={{ flex: 1, resize: "none", minHeight: "150px" }}
      />
    </div>
  );
}

export default BenchmarkResult;
