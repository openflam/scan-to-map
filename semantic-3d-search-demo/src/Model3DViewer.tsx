// import "./App.css";
import { useEffect, useRef } from "react";
import { Viewer, CreateViewerForCanvas } from "@babylonjs/viewer";
import { MeshBuilder, Color3, StandardMaterial } from "@babylonjs/core";
import type { BoundingBox } from "./types/global";

function Model3DViewer(props: {
  source: string | ArrayBufferView | File;
  boundingBox?: BoundingBox;
  autoTagBBoxes?: BoundingBox[];
  showAutoTags?: boolean;
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

    return () => { };
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
      createBoundingBoxMesh(props.boundingBox, sceneRef.current, Color3.Red(), 0.1);

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

  // Update auto tag bounding boxes separately
  useEffect(() => {
    if (!sceneRef.current || !viewerRef.current) return;

    // Remove existing auto tag bounding box meshes if they exist
    const existingAutoTagMeshes = sceneRef.current.meshes.filter(
      (mesh: any) => mesh.name && mesh.name.startsWith("autoTagBox_")
    );
    existingAutoTagMeshes.forEach((mesh: any) => mesh.dispose());

    // Create new auto tag bounding boxes if provided and enabled
    if (props.showAutoTags && props.autoTagBBoxes && props.autoTagBBoxes.length > 0) {
      console.log("Updating auto tag bounding boxes:", props.autoTagBBoxes.length);

      props.autoTagBBoxes.forEach((bbox, index) => {
        // Generate a random color for each box
        const randomColor = new Color3(Math.random(), Math.random(), Math.random());
        createBoundingBoxMesh(
          bbox,
          sceneRef.current,
          randomColor,
          0.05,
          `autoTagBox_${index}`
        );
      });

      // Request a safe render through the viewer's engine
      if (viewerRef.current && sceneRef.current) {
        const engine = sceneRef.current.getEngine();
        if (engine && !engine.isDisposed) {
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
  }, [props.autoTagBBoxes, props.showAutoTags]); // Re-run when auto tag boxes or visibility changes

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

// Function to create bounding box mesh given bounding box coordinates
function createBoundingBoxMesh(
  boundingBox: BoundingBox,
  sceneRef: any,
  color: Color3 = Color3.Red(),
  alpha: number = 0.1,
  name: string = "boundingBox"
) {
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
    name,
    { width, height, depth },
    sceneRef
  );

  box.position.set(centerX, centerY, centerZ);

  // Create semi-transparent material with specified color
  const material = new StandardMaterial(`${name}Material`, sceneRef);
  material.diffuseColor = color;
  material.emissiveColor = color;
  material.alpha = alpha;
  material.backFaceCulling = false; // render both sides for better visibility

  // Create a slightly larger wireframe overlay to show edges in darker shade
  const edgeMaterial = new StandardMaterial(
    `${name}EdgeMaterial`,
    sceneRef
  );
  const darkerColor = color.scale(0.45); // darker version of the color
  edgeMaterial.diffuseColor = darkerColor;
  edgeMaterial.emissiveColor = darkerColor;
  edgeMaterial.wireframe = true;
  edgeMaterial.backFaceCulling = false;

  // Clone the box to draw the edges on top and avoid z-fighting by scaling slightly
  const edgeBox = box.clone(`${name}Edges`);
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
