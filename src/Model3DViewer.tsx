// import "./App.css";
import { useEffect, useRef } from "react";
import { Viewer, CreateViewerForCanvas } from "@babylonjs/viewer";
import { MeshBuilder, Color3, StandardMaterial } from "@babylonjs/core";
import type { BoundingBox } from "./types/global";

function Model3DViewer(props: {
  source: string | ArrayBufferView | File;
  boundingBox?: BoundingBox;
}) {
  const canvasRef = useRef(null);
  const viewerRef = useRef<Viewer | null>(null);
  const sceneRef = useRef<any>(null);

  // Initialize viewer only once
  useEffect(() => {
    if (!canvasRef.current) return;

    const viewerPromise = CreateViewerForCanvas(canvasRef.current, {
      engine: "WebGPU",
      onInitialized: (details) => {
        sceneRef.current = details.scene;
      },
    });

    viewerPromise.then((viewer) => {
      viewer.loadModel(props.source);

      viewer.onModelChanged.add(() => {
        viewerRef.current = viewer;
      });
    });

    return () => {};
  }, [props.source]); // Only re-run when source changes

  // Update bounding box separately
  useEffect(() => {
    if (!sceneRef.current || !viewerRef.current) return;

    // Remove existing bounding box meshes if they exist
    const existingBox = sceneRef.current.getMeshByName("boundingBox");
    const existingEdges = sceneRef.current.getMeshByName("boundingBoxEdges");

    if (existingBox) {
      existingBox.dispose();
    }
    if (existingEdges) {
      existingEdges.dispose();
    }

    // Create new bounding box if provided
    if (props.boundingBox) {
      console.log("Updating bounding box:", props.boundingBox);
      createBoundingBoxMesh(props.boundingBox, sceneRef.current);

      // Request a safe render through the viewer's engine
      // This avoids the WebGPU destroyed texture error
      if (viewerRef.current && sceneRef.current) {
        const engine = sceneRef.current.getEngine();
        if (engine && !engine.isDisposed) {
          // Schedule render on next frame safely
          requestAnimationFrame(() => {
            if (engine && !engine.isDisposed) {
              engine.stopRenderLoop();
              engine.runRenderLoop(() => {
                if (sceneRef.current && !engine.isDisposed) {
                  sceneRef.current.render();
                }
              });
            }
          });
        }
      }
    }
  }, [props.boundingBox]); // Only re-run when bounding box changes

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

// Function to create bounding box mesh given bounding box coordinates
function createBoundingBoxMesh(boundingBox: BoundingBox, sceneRef: any) {
  const { x_min, y_min, z_min, x_max, y_max, z_max } = boundingBox;

  // Calculate center and size
  const centerX = (x_min + x_max) / 2;
  const centerY = (y_min + y_max) / 2;
  const centerZ = (z_min + z_max) / 2;

  const width = x_max - x_min;
  const height = y_max - y_min;
  const depth = z_max - z_min;

  // Create box mesh
  const box = MeshBuilder.CreateBox(
    "boundingBox",
    { width, height, depth },
    sceneRef
  );

  box.position.set(centerX, centerY, centerZ);

  // Create red semi-transparent material
  const material = new StandardMaterial("boundingBoxMaterial", sceneRef);
  material.diffuseColor = Color3.Red();
  material.emissiveColor = Color3.Red();
  material.alpha = 0.1;
  material.backFaceCulling = false; // render both sides for better visibility

  // Create a slightly larger wireframe overlay to show edges in dark red
  const edgeMaterial = new StandardMaterial(
    "boundingBoxEdgeMaterial",
    sceneRef
  );
  edgeMaterial.diffuseColor = new Color3(0.45, 0, 0); // dark red
  edgeMaterial.emissiveColor = new Color3(0.45, 0, 0);
  edgeMaterial.wireframe = true;
  edgeMaterial.backFaceCulling = false;

  // Clone the box to draw the edges on top and avoid z-fighting by scaling slightly
  const edgeBox = box.clone("boundingBoxEdges");
  if (edgeBox) {
    edgeBox.material = edgeMaterial;
    // copy and slightly scale up to prevent z-fighting with the translucent box
    edgeBox.scaling = box.scaling.clone ? box.scaling.clone() : box.scaling;
    edgeBox.scaling.scaleInPlace(1.002);
    edgeBox.isPickable = false;
  }

  box.material = material;
}

export default Model3DViewer;
