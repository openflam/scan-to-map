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
  Vector3,
} from "@babylonjs/core";
import type { BoundingBox, Route } from "./types/global";

function Model3DViewer(props: {
  source: string | ArrayBufferView | File;
  boundingBox?: BoundingBox[];
  autoTagBBoxes?: BoundingBox[];
  showAutoTags?: boolean;
  occupancyGrid?: BoundingBox[];
  showOccupancyGrid?: boolean;
  annotations?: string[];
  route?: Route;
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

  // Update route separately
  useEffect(() => {
    if (!sceneRef.current || !viewerRef.current) return;

    // Remove existing route meshes if they exist
    const existingRouteMeshes = sceneRef.current.meshes.filter(
      (mesh: any) => mesh.name && mesh.name.startsWith("routeSegment_")
    );
    existingRouteMeshes.forEach((mesh: any) => mesh.dispose());

    // Create new route if provided
    if (props.route && props.route.length > 1) {
      console.log("Rendering route with", props.route.length, "points");

      const blueColor = new Color3(0, 0, 1);

      // Create cylinders connecting adjacent points
      for (let i = 0; i < props.route.length - 1; i++) {
        const start = props.route[i];
        const end = props.route[i + 1];

        createRouteCylinder(
          start,
          end,
          sceneRef.current,
          blueColor,
          `routeSegment_${i}`
        );
      }

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
  }, [props.route]); // Re-run when route changes

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

// Function to create a cylinder connecting two points for route visualization
function createRouteCylinder(
  start: [number, number, number],
  end: [number, number, number],
  sceneRef: any,
  color: Color3 = Color3.Blue(),
  name: string = "routeSegment"
) {
  const [x1, y1, z1] = start;
  const [x2, y2, z2] = end;

  // Calculate distance between points
  const dx = x2 - x1;
  const dy = y2 - y1;
  const dz = z2 - z1;
  const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

  // Create cylinder with small diameter (thin line)
  const diameter = 0.1;
  const cylinder = MeshBuilder.CreateCylinder(
    name,
    {
      height: distance,
      diameter: diameter,
    },
    sceneRef
  );

  // Position cylinder at midpoint
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2;
  const midZ = (z1 + z2) / 2;
  cylinder.position.set(midX, midY, midZ);

  // Rotate cylinder to align with the direction vector
  // Calculate rotation to align cylinder (default Y-axis) with direction vector
  const direction = { x: dx, y: dy, z: dz };
  const dirLength = distance;

  if (dirLength > 0.0001) {
    // Normalize direction
    const dirNorm = {
      x: direction.x / dirLength,
      y: direction.y / dirLength,
      z: direction.z / dirLength,
    };

    // Calculate rotation axis (cross product of Y-axis with direction)
    const yAxis = { x: 0, y: 1, z: 0 };
    const rotationAxis = {
      x: yAxis.y * dirNorm.z - yAxis.z * dirNorm.y,
      y: yAxis.z * dirNorm.x - yAxis.x * dirNorm.z,
      z: yAxis.x * dirNorm.y - yAxis.y * dirNorm.x,
    };

    // Calculate rotation angle (dot product)
    const dotProduct =
      yAxis.x * dirNorm.x + yAxis.y * dirNorm.y + yAxis.z * dirNorm.z;
    const angle = Math.acos(Math.max(-1, Math.min(1, dotProduct)));

    // Apply rotation if needed
    if (Math.abs(angle) > 0.0001) {
      const axisLength = Math.sqrt(
        rotationAxis.x * rotationAxis.x +
          rotationAxis.y * rotationAxis.y +
          rotationAxis.z * rotationAxis.z
      );

      if (axisLength > 0.0001) {
        cylinder.rotate(
          new Vector3(
            rotationAxis.x / axisLength,
            rotationAxis.y / axisLength,
            rotationAxis.z / axisLength
          ),
          angle
        );
      }
    }
  }

  // Create material
  const material = new StandardMaterial(`${name}Material`, sceneRef);
  material.diffuseColor = color;
  material.emissiveColor = color;
  material.alpha = 1.0;
  material.backFaceCulling = false;

  cylinder.material = material;
  // cylinder.renderingGroupId = 1; // Render in front
}

export default Model3DViewer;
