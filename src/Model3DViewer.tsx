// import "./App.css";
import { useEffect, useRef } from "react";
import { Viewer, CreateViewerForCanvas } from "@babylonjs/viewer";

function Model3DViewer(props: { source: string | ArrayBufferView | File }) {
  const canvasRef = useRef(null);
  const viewerRef = useRef<Viewer | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const viewerPromise = CreateViewerForCanvas(canvasRef.current, {
      engine: "WebGPU",
      onInitialized: (details) => {
        console.log("DETAILS", details);
      },
    });

    viewerPromise.then((viewer) => {
      viewer.loadModel(props.source);

      viewer.onModelChanged.add(() => {
        console.log("Model changed");
        viewerRef.current = viewer;
      });
    });

    return () => {};
  }, [canvasRef]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

export default Model3DViewer;
