// import "./App.css";
import { useEffect, useRef } from "react";
import { Viewer, CreateViewerForCanvas } from "@babylonjs/viewer";
import {
  MeshBuilder,
  Color3,
  StandardMaterial,
  DynamicTexture,
  Mesh,
  ArcRotateCamera,
} from "@babylonjs/core";
import type { BoundingBox } from "./types/global";

function Model3DViewer(props: {
  source: string | ArrayBufferView | File;
  boundingBox?: BoundingBox[];
  autoTagBBoxes?: BoundingBox[];
  showAutoTags?: boolean;
  occupancyGrid?: BoundingBox[];
  showOccupancyGrid?: boolean;
  annotations?: string[];
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

        const scene = sceneRef.current;
        if (!scene) return;

        const cam = scene.activeCamera as ArcRotateCamera | null;
        if (!cam) return;

        // // Stop percentage-based zoom so it doesn't approach 0 near the model
        cam.wheelDeltaPercentage = 0;
        cam.pinchDeltaPercentage = 0;

        // Constant zoom step tuning (smaller numbers = faster)
        cam.wheelPrecision = 0.5; // try 1–2; raise if too fast
        cam.pinchPrecision = 20;

        // Let the camera get very close and avoid near-plane clipping
        cam.lowerRadiusLimit = 0.00001;
        cam.minZ = 0.001;

        // Nice-to-have
        cam.zoomToMouseLocation = true;
      });
    });

    return () => {};
  }, [props.source]); // Only re-run when source changes

  // Update bounding box separately
  useEffect(() => {
    if (!sceneRef.current || !viewerRef.current) return;

    // Remove existing bounding box meshes if they exist
    const existingBBoxMeshes = sceneRef.current.meshes.filter(
      (mesh: any) =>
        mesh.name &&
        (mesh.name.startsWith("boundingBox") ||
          mesh.name === "boundingBoxEdges")
    );
    existingBBoxMeshes.forEach((mesh: any) => mesh.dispose());

    // Create new bounding box(es) if provided
    if (props.boundingBox && props.boundingBox.length > 0) {
      console.log("Updating bounding box(es):", props.boundingBox.length);

      props.boundingBox.forEach((bbox, index) => {
        const name =
          props.boundingBox!.length > 1
            ? `boundingBox_${index}`
            : "boundingBox";
        createBoundingBoxMesh(bbox, sceneRef.current, Color3.Red(), 0.3, name);
      });

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
      (mesh: any) =>
        mesh.name &&
        (mesh.name.startsWith("autoTagBox_") ||
          mesh.name.startsWith("autoTagLabel_"))
    );
    existingAutoTagMeshes.forEach((mesh: any) => mesh.dispose());

    // Create new auto tag bounding boxes if provided and enabled
    if (
      props.showAutoTags &&
      props.autoTagBBoxes &&
      props.autoTagBBoxes.length > 0
    ) {
      console.log(
        "Updating auto tag bounding boxes:",
        props.autoTagBBoxes.length
      );

      props.autoTagBBoxes.forEach((bbox, index) => {
        // Generate a random color for each box
        const randomColor = new Color3(
          Math.random(),
          Math.random(),
          Math.random()
        );
        createBoundingBoxMesh(
          bbox,
          sceneRef.current,
          randomColor,
          0.05,
          `autoTagBox_${index}`,
          false // Don't render in front
        );

        // Create label if annotation exists
        if (props.annotations && props.annotations[index]) {
          const label = props.annotations[index];
          createTextLabel(
            bbox,
            sceneRef.current,
            label,
            `autoTagLabel_${index}`
          );
        }
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
  }, [props.autoTagBBoxes, props.showAutoTags, props.annotations]); // Re-run when auto tag boxes or visibility changes

  // Update occupancy grid bounding boxes separately
  useEffect(() => {
    if (!sceneRef.current || !viewerRef.current) return;

    // Remove existing occupancy grid bounding box meshes if they exist
    const existingOccupancyMeshes = sceneRef.current.meshes.filter(
      (mesh: any) => mesh.name && mesh.name.startsWith("occupancyBox_")
    );
    existingOccupancyMeshes.forEach((mesh: any) => mesh.dispose());

    // Create new occupancy grid bounding boxes if provided and enabled
    if (
      props.showOccupancyGrid &&
      props.occupancyGrid &&
      props.occupancyGrid.length > 0
    ) {
      console.log(
        "Updating occupancy grid bounding boxes:",
        props.occupancyGrid.length
      );

      props.occupancyGrid.forEach((bbox, index) => {
        // Use green color for all occupancy grid boxes
        const greenColor = new Color3(0, 1, 0);
        createBoundingBoxMesh(
          bbox,
          sceneRef.current,
          greenColor,
          0.05,
          `occupancyBox_${index}`,
          false // Don't render in front
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
  }, [props.occupancyGrid, props.showOccupancyGrid]); // Re-run when occupancy grid or visibility changes

  return <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />;
}

// Function to create bounding box mesh given bounding box coordinates
function createBoundingBoxMesh(
  boundingBox: BoundingBox,
  sceneRef: any,
  color: Color3 = Color3.Red(),
  alpha: number = 0.1,
  name: string = "boundingBox",
  inFront: boolean = true
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
  const box = MeshBuilder.CreateBox(name, { width, height, depth }, sceneRef);

  box.position.set(centerX, centerY, centerZ);

  // Create semi-transparent material with specified color
  const material = new StandardMaterial(`${name}Material`, sceneRef);
  material.diffuseColor = color;
  material.emissiveColor = color;
  material.alpha = alpha;
  material.backFaceCulling = false; // render both sides for better visibility

  // Create a slightly larger wireframe overlay to show edges in darker shade
  const edgeMaterial = new StandardMaterial(`${name}EdgeMaterial`, sceneRef);
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

  // Make the bounding box always render on top (only if inFront is true)
  if (inFront) {
    box.renderingGroupId = 1;
    if (edgeBox) {
      edgeBox.renderingGroupId = 1;
    }
  }
}

// Function to create text label for bounding box
function createTextLabel(
  boundingBox: BoundingBox,
  sceneRef: any,
  text: string,
  name: string = "textLabel"
) {
  const { x_min, y_min: _y_min, z_min, x_max, y_max, z_max } = boundingBox;

  // Calculate center position (place label at top center of bounding box)
  const centerX = (x_min + x_max) / 2;
  const centerY = y_max + 0.1; // Top of the box
  const centerZ = (z_min + z_max) / 2;

  // Create a plane for the text
  const plane = MeshBuilder.CreatePlane(
    name,
    { width: 0.2, height: 0.2 },
    sceneRef
  );
  plane.position.set(centerX, centerY, centerZ);

  // Make the plane always face the camera
  plane.billboardMode = Mesh.BILLBOARDMODE_ALL;

  // Create dynamic texture for text
  const textureResolution = 512;
  const texture = new DynamicTexture(
    `${name}Texture`,
    textureResolution,
    sceneRef,
    false
  );

  const ctx = texture.getContext() as CanvasRenderingContext2D;
  const fontSize = 100;

  // Clear and set background
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.fillRect(0, 0, textureResolution, textureResolution);

  // Draw text
  ctx.font = `bold ${fontSize}px Arial`;
  ctx.fillStyle = "white";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, textureResolution / 2, textureResolution / 2);

  texture.update();

  // Create material and apply texture
  const material = new StandardMaterial(`${name}Material`, sceneRef);
  material.diffuseTexture = texture;
  material.emissiveTexture = texture;
  material.opacityTexture = texture;
  material.backFaceCulling = false;

  plane.material = material;
}

export default Model3DViewer;
