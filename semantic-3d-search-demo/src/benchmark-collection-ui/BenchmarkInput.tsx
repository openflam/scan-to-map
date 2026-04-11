import { Form, Button, Accordion } from "react-bootstrap";
import { useState, useEffect } from "react";
import { parseResult } from "../SearchResult";
import {
  callTool,
  saveBenchmark,
  getBenchmarkNames,
  getBenchmark,
} from "../query";

interface BenchmarkInputProps {
  componentIds?: string[];
  onComponentClick?: (index: number) => void;
  datasetName: string;
}

function BenchmarkInput({
  componentIds,
  onComponentClick,
  datasetName,
}: BenchmarkInputProps) {
  const [question, setQuestion] = useState("");
  const [expectedAnswer, setExpectedAnswer] = useState("");
  const [isPreview, setIsPreview] = useState(false);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [benchmarkType, setBenchmarkType] = useState("Entity Search");
  const [benchmarkNamesList, setBenchmarkNamesList] = useState<string[]>([]);

  useEffect(() => {
    getBenchmarkNames().then(setBenchmarkNamesList).catch(console.error);
  }, []);

  const handleNameChange = async (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    const val = e.target.value;
    setBenchmarkName(val);
    if (benchmarkNamesList.includes(val)) {
      try {
        const data = await getBenchmark(val);
        if (data.question !== undefined) setQuestion(data.question);
        if (data.expected_answer !== undefined)
          setExpectedAnswer(data.expected_answer);
        if (data.benchmark_type !== undefined)
          setBenchmarkType(data.benchmark_type);
      } catch (err) {
        console.error(err);
      }
    }
  };

  // Distance parameters
  const [distId1, setDistId1] = useState("");
  const [distId2, setDistId2] = useState("");
  const [distanceResult, setDistanceResult] = useState<string | null>(null);
  const [isGettingDistance, setIsGettingDistance] = useState(false);

  const handleGetDistance = async () => {
    if (!distId1 || !distId2) return;
    setIsGettingDistance(true);
    setDistanceResult(null);
    try {
      const resp = await callTool(
        "get_distance",
        {
          component_id_1: parseInt(distId1),
          component_id_2: parseInt(distId2),
        },
        datasetName,
      );

      if (resp.result && resp.result.error) {
        setDistanceResult(`Error: ${resp.result.error}`);
      } else if (resp.result !== undefined) {
        setDistanceResult(`${Number(resp.result).toFixed(2)} meters`);
      } else {
        setDistanceResult("Unknown error");
      }
    } catch (e) {
      setDistanceResult("API Error");
    } finally {
      setIsGettingDistance(false);
    }
  };

  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await saveBenchmark(
        datasetName,
        question,
        expectedAnswer,
        benchmarkName,
        benchmarkType,
      );
      setQuestion("");
      setExpectedAnswer("");
      setBenchmarkName("");
    } catch (e) {
      console.error(e);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div
      className="d-flex flex-column h-100 pb-2"
      style={{ overflowY: "auto" }}
    >
      {/* Tools UI */}
      <Accordion className="mb-3">
        <Accordion.Item eventKey="0">
          <Accordion.Header>
            <span className="text-secondary fw-bold">Tools</span>
          </Accordion.Header>
          <Accordion.Body>
            <div className="d-flex flex-column">
              <div className="d-flex justify-content-between align-items-center mb-1">
                <span className="text-secondary fw-bold">Get Distance</span>
              </div>
              <div className="d-flex gap-2 align-items-center">
                <Form.Control
                  placeholder="ID 1"
                  value={distId1}
                  onChange={(e) => setDistId1(e.target.value)}
                  size="sm"
                />
                <Form.Control
                  placeholder="ID 2"
                  value={distId2}
                  onChange={(e) => setDistId2(e.target.value)}
                  size="sm"
                />
                <Button
                  size="sm"
                  variant="outline-primary"
                  onClick={handleGetDistance}
                  disabled={isGettingDistance}
                  style={{ whiteSpace: "nowrap" }}
                >
                  Calculate
                </Button>
              </div>
              {distanceResult && (
                <div
                  className="text-muted mt-2"
                  style={{ fontSize: "0.95rem" }}
                >
                  Result: <span className="fw-bold">{distanceResult}</span>
                </div>
              )}
            </div>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>

      <div className="d-flex flex-column mb-3" style={{ flex: "0 0 auto" }}>
        <div className="d-flex justify-content-between align-items-center mb-1">
          <span className="text-secondary fw-bold">Name</span>
        </div>
        <Form.Control
          placeholder="Benchmark Name"
          value={benchmarkName}
          onChange={handleNameChange}
          list="benchmark-names"
        />
        <datalist id="benchmark-names">
          {benchmarkNamesList.map((name) => (
            <option key={name} value={name} />
          ))}
        </datalist>
      </div>

      <div className="d-flex flex-column mb-3" style={{ flex: "0 0 auto" }}>
        <div className="d-flex justify-content-between align-items-center mb-1">
          <span className="text-secondary fw-bold">Type</span>
        </div>
        <Form.Select
          value={benchmarkType}
          onChange={(e) => setBenchmarkType(e.target.value)}
        >
          <option value="Entity Search">Entity Search</option>
          <option value="Spatial Relations">Spatial Relations</option>
          <option value="Affordance">Affordance</option>
          <option value="Functionality">Functionality</option>
          <option value="Physics, safety, etc.">Physics, safety, etc.</option>
        </Form.Select>
      </div>

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
        <Button variant="primary" onClick={handleSave} disabled={isSaving}>
          {isSaving ? "Saving..." : "Save"}
        </Button>
      </div>
    </div>
  );
}

export default BenchmarkInput;
