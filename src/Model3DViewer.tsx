// import "./App.css";
import { useEffect, useRef } from "react";
import { Viewer, CreateViewerForCanvas } from "@babylonjs/viewer";
import { MeshBuilder, Color3, StandardMaterial } from "@babylonjs/core";

interface BoundingBox {
  x_min: number;
  y_min: number;
  z_min: number;
  x_max: number;
  y_max: number;
  z_max: number;
}

function Model3DViewer(props: {
  source: string | ArrayBufferView | File;
  boundingBox?: BoundingBox;
}) {
  const canvasRef = useRef(null);
  const viewerRef = useRef<Viewer | null>(null);

  // TEMP FOR TEST -- Function to calculate bounding box from scene meshes
  const calculateBoundingBoxFromScene = (scene: any): BoundingBox | null => {
    const meshes = scene.meshes.filter(
      (mesh: any) => mesh.getTotalVertices() > 0
    );

    if (meshes.length === 0) {
      return null;
    }

    // Initialize with extreme values
    let x_min = Infinity;
    let y_min = Infinity;
    let z_min = Infinity;
    let x_max = -Infinity;
    let y_max = -Infinity;
    let z_max = -Infinity;

    // Calculate overall bounding box from all meshes
    meshes.forEach((mesh: any) => {
      const boundingInfo = mesh.getBoundingInfo();
      const min = boundingInfo.boundingBox.minimumWorld;
      const max = boundingInfo.boundingBox.maximumWorld;

      x_min = Math.min(x_min, min.x);
      y_min = Math.min(y_min, min.y);
      z_min = Math.min(z_min, min.z);
      x_max = Math.max(x_max, max.x);
      y_max = Math.max(y_max, max.y);
      z_max = Math.max(z_max, max.z);
    });

    const boundingBox = { x_min, y_min, z_min, x_max, y_max, z_max };
    console.log("Calculated bounding box:", boundingBox);
    return boundingBox;
  };

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

  useEffect(() => {
    if (!canvasRef.current) return;

    let sceneRef: any = null;

    const viewerPromise = CreateViewerForCanvas(canvasRef.current, {
      engine: "WebGPU",
      onInitialized: (details) => {
        sceneRef = details.scene;
      },
    });

    viewerPromise.then((viewer) => {
      viewer.loadModel(props.source);

      viewer.onModelChanged.add(() => {
        console.log("Model changed");
        viewerRef.current = viewer;

        if (sceneRef) {
          // Get bounding box (either from props or calculated from scene)
          const boundingBox =
            props.boundingBox || calculateBoundingBoxFromScene(sceneRef);

          if (!boundingBox) {
            // No meshes found, skip bounding box
            return;
          }

          createBoundingBoxMesh(boundingBox, sceneRef);
        }
      });
    });

    return () => {};
  }, [canvasRef, props.source, props.boundingBox]);

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

export default Model3DViewer;
