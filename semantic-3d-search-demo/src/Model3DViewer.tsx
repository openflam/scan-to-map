import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Text,
  Line,
  Edges,
  Billboard,
  KeyboardControls,
  useKeyboardControls,
} from "@react-three/drei";
import { Suspense, useMemo, useState, useEffect } from "react";
import * as THREE from "three";
import type { BoundingBox, Route } from "./types/global";

interface Model3DViewerProps {
  source: string;
  boundingBox?: BoundingBox[];
  captions?: string[];
  componentIds?: string[];
  autoTagBBoxes?: BoundingBox[];
  showAutoTags?: boolean;
  occupancyGrid?: BoundingBox[];
  showOccupancyGrid?: boolean;
  annotations?: string[];
  route?: Route;
}

function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} />;
}

function BoundingBoxMesh({
  bbox,
  color,
  label,
  opacity = 0.1,
  onClick,
  isSelected = false,
  isDimmed = false,
}: {
  bbox: BoundingBox;
  color: string | THREE.Color;
  label?: string;
  opacity?: number;
  onClick?: (e: any) => void;
  isSelected?: boolean;
  isDimmed?: boolean;
}) {
  const { x_min, y_min, z_min, x_max, y_max, z_max } = bbox;
  const width = Math.abs(x_max - x_min);
  const height = Math.abs(y_max - y_min);
  const depth = Math.abs(z_max - z_min);
  const position: [number, number, number] = [
    (x_min + x_max) / 2,
    (y_min + y_max) / 2,
    (z_min + z_max) / 2,
  ];

  const finalOpacity = isDimmed ? 0.05 : isSelected ? 0.6 : opacity;
  const finalColor = isSelected ? "#3b82f6" : color;

  return (
    <group position={position}>
      <mesh
        onClick={(e) => {
          if (e.button === 0) {
            e.stopPropagation();
            if (onClick) onClick(e);
          }
        }}
      >
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial
          color={finalColor}
          transparent
          opacity={finalOpacity}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
        <Edges color={new THREE.Color(finalColor).multiplyScalar(0.5)} />
      </mesh>
      {label && !isDimmed && (
        <Billboard position={[0, height / 2 + 0.1, 0]}>
          <Text
            fontSize={0.1}
            color="white"
            anchorX="center"
            anchorY="bottom"
            outlineWidth={0.02}
            outlineColor="black"
          >
            {label}
          </Text>
        </Billboard>
      )}
    </group>
  );
}

function CameraController() {
  const [, get] = useKeyboardControls();
  const { camera } = useThree();
  useFrame((state, delta) => {
    const activeElement = document.activeElement;
    if (activeElement && ["INPUT", "TEXTAREA"].includes(activeElement.tagName))
      return;
    const { forward, backward, left, right } = get();
    if (!forward && !backward && !left && !right) return;
    const speed = 5 * delta;
    const direction = new THREE.Vector3();
    if (forward) direction.z -= 1;
    if (backward) direction.z += 1;
    if (left) direction.x -= 1;
    if (right) direction.x += 1;
    if (direction.lengthSq() === 0) return;
    direction
      .normalize()
      .multiplyScalar(speed)
      .applyQuaternion(camera.quaternion);
    camera.position.add(direction);
    const controls = state.controls as any;
    if (controls) controls.target.add(direction);
  });
  return null;
}

function GlobalInputHandler({ onExit }: { onExit: () => void }) {
  const [, get] = useKeyboardControls();
  useFrame(() => {
    if (get().escape) onExit();
  });
  return null;
}

function ComponentDetails({
  editedCaption,
  setEditedCaption,
  isEditing,
  setIsEditing,
  onDismiss,
  componentId,
  imageBase64,
  isLoading,
}: {
  editedCaption: string;
  setEditedCaption: (value: string) => void;
  isEditing: boolean;
  setIsEditing: (value: boolean) => void;
  onDismiss: () => void;
  componentId: string | null;
  imageBase64?: string | null;
  isLoading?: boolean;
}) {
  const handleSave = async () => {
    setIsEditing(false);
    if (!componentId) return;
    try {
      const { updateComponentCaption } = await import("./query");
      await updateComponentCaption(componentId, editedCaption);
    } catch (error) {
      console.error("Error saving caption:", error);
    }
  };
  return (
    <div
      style={{
        position: "absolute",
        top: "20px",
        right: "20px",
        width: "300px",
        maxHeight: "calc(100% - 40px)",
        overflowY: "auto",
        zIndex: 10,
        backgroundColor: "rgba(255, 255, 255, 0.95)",
        padding: "20px",
        borderRadius: "8px",
        boxShadow: "0 10px 25px rgba(0,0,0,0.15)",
        border: "1px solid #eee",
        display: "flex",
        flexDirection: "column",
        gap: "12px",
        backdropFilter: "blur(4px)",
      }}
    >
      <h3
        style={{
          margin: 0,
          fontSize: "12px",
          color: "#888",
          letterSpacing: "1px",
          textTransform: "uppercase",
        }}
      >
        Annotation Detail
      </h3>

      {isLoading ? (
        <div style={{ textAlign: "center", padding: "20px" }}>
          <p style={{ color: "#6b7280" }}>Loading...</p>
        </div>
      ) : (
        <>
          {imageBase64 && (
            <div
              style={{
                width: "100%",
                borderRadius: "6px",
                overflow: "hidden",
                marginBottom: "8px",
              }}
            >
              <img
                src={`data:image/jpeg;base64,${imageBase64}`}
                alt="Component view"
                style={{
                  width: "100%",
                  height: "auto",
                  display: "block",
                }}
              />
            </div>
          )}

          {isEditing ? (
            <textarea
              value={editedCaption}
              onChange={(e) => setEditedCaption(e.target.value)}
              style={{
                width: "100%",
                height: "240px",
                padding: "10px",
                borderRadius: "6px",
                border: "2px solid #3b82f6",
                fontSize: "14px",
                outline: "none",
                resize: "none",
                overflowY: "auto",
              }}
              autoFocus
            />
          ) : (
            <div
              style={{
                minHeight: "60px",
                maxHeight: "240px",
                overflowY: "auto",
                padding: "10px",
                borderRadius: "6px",
                border: "1px solid #e5e7eb",
              }}
            >
              <p
                style={{
                  margin: 0,
                  fontSize: "16px",
                  color: "#1f2937",
                  lineHeight: "1.5",
                }}
              >
                {editedCaption || "No caption available."}
              </p>
            </div>
          )}
        </>
      )}

      <div style={{ display: "flex", gap: "10px" }}>
        <button
          onClick={() => setIsEditing(!isEditing)}
          style={{
            flex: 1,
            padding: "10px",
            cursor: "pointer",
            backgroundColor: isEditing ? "#fee2e2" : "#f3f4f6",
            color: isEditing ? "#ef4444" : "#374151",
            border: "1px solid #d1d5db",
            borderRadius: "6px",
            fontSize: "14px",
            fontWeight: "600",
            transition: "all 0.2s",
          }}
        >
          {isEditing ? "Cancel" : "Edit"}
        </button>
        <button
          onClick={handleSave}
          style={{
            flex: 1,
            padding: "10px",
            cursor: "pointer",
            backgroundColor: "#3b82f6",
            color: "white",
            border: "none",
            borderRadius: "6px",
            fontSize: "14px",
            fontWeight: "600",
            transition: "background-color 0.2s",
          }}
        >
          Save
        </button>
      </div>
      <button
        onClick={onDismiss}
        style={{
          background: "none",
          border: "none",
          color: "#9ca3af",
          fontSize: "12px",
          cursor: "pointer",
          alignSelf: "center",
          marginTop: "4px",
        }}
      >
        Dismiss (Esc)
      </button>
    </div>
  );
}

export default function Model3DViewer({
  source,
  boundingBox,
  captions,
  componentIds,
  autoTagBBoxes,
  showAutoTags,
  occupancyGrid,
  showOccupancyGrid,
  annotations,
  route,
}: Model3DViewerProps) {
  const [selectedBBoxIndex, setSelectedBBoxIndex] = useState<number | null>(
    null,
  );
  const [selectedAutoTagId, setSelectedAutoTagId] = useState<string | null>(
    null,
  );
  const [isEditing, setIsEditing] = useState(false);
  const [editedCaption, setEditedCaption] = useState("");
  const [componentImage, setComponentImage] = useState<string | null>(null);
  const [isLoadingComponent, setIsLoadingComponent] = useState(false);

  const currentComponentId: string | null = useMemo(() => {
    if (selectedAutoTagId !== null) return selectedAutoTagId;
    if (selectedBBoxIndex !== null && componentIds)
      return componentIds[selectedBBoxIndex] ?? null;
    return null;
  }, [selectedAutoTagId, selectedBBoxIndex, componentIds]);

  useEffect(() => {
    const fetchComponentInfo = async () => {
      // Handle autotag selection
      if (selectedAutoTagId !== null) {
        setIsLoadingComponent(true);
        try {
          const { getComponentInfo } = await import("./query");
          const info = await getComponentInfo(selectedAutoTagId);
          setEditedCaption(info.caption || "");
          setComponentImage(info.image_base64);
        } catch (error) {
          console.error("Error fetching component info:", error);
          setEditedCaption("Error loading component information");
          setComponentImage(null);
        } finally {
          setIsLoadingComponent(false);
        }
      }
      // Handle search result selection
      else if (selectedBBoxIndex !== null && componentIds) {
        const componentId = componentIds[selectedBBoxIndex];
        if (componentId) {
          setIsLoadingComponent(true);
          try {
            const { getComponentInfo } = await import("./query");
            const info = await getComponentInfo(componentId);
            setEditedCaption(info.caption || "");
            setComponentImage(info.image_base64);
          } catch (error) {
            console.error("Error fetching component info:", error);
            // Fallback to captions array if API fails
            if (captions) {
              setEditedCaption(captions[selectedBBoxIndex] || "");
            }
            setComponentImage(null);
          } finally {
            setIsLoadingComponent(false);
          }
        } else if (captions) {
          setEditedCaption(captions[selectedBBoxIndex] || "");
          setComponentImage(null);
        }
      }
      setIsEditing(false);
    };

    fetchComponentInfo();
  }, [selectedBBoxIndex, selectedAutoTagId, captions, componentIds]);

  const map = useMemo(
    () => [
      { name: "forward", keys: ["ArrowUp", "KeyW"] },
      { name: "backward", keys: ["ArrowDown", "KeyS"] },
      { name: "left", keys: ["ArrowLeft", "KeyA"] },
      { name: "right", keys: ["ArrowRight", "KeyD"] },
      { name: "escape", keys: ["Escape"] },
    ],
    [],
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
        overflow: "hidden",
        fontFamily: "sans-serif",
      }}
    >
      <KeyboardControls map={map}>
        {/* --- HTML OVERLAY --- */}
        {(selectedBBoxIndex !== null || selectedAutoTagId !== null) && (
          <ComponentDetails
            editedCaption={editedCaption}
            setEditedCaption={setEditedCaption}
            isEditing={isEditing}
            setIsEditing={setIsEditing}
            onDismiss={() => {
              setSelectedBBoxIndex(null);
              setSelectedAutoTagId(null);
            }}
            componentId={currentComponentId}
            imageBase64={componentImage}
            isLoading={isLoadingComponent}
          />
        )}

        {/* --- 3D SCENE --- */}
        <Canvas
          camera={{ position: [5, 5, 5] }}
          /* Removed onPointerMissed to prevent deselection on background click */
        >
          <ambientLight intensity={2.0} />
          <pointLight position={[10, 10, 10]} intensity={2.0} />
          <OrbitControls makeDefault />
          <CameraController />
          {/* Preserved Esc Key logic here */}
          <GlobalInputHandler
            onExit={() => {
              setSelectedBBoxIndex(null);
              setSelectedAutoTagId(null);
            }}
          />

          <Suspense fallback={null}>
            <Model url={source} />
          </Suspense>

          {boundingBox?.map((bbox, i) => (
            <BoundingBoxMesh
              key={`bbox-${i}`}
              bbox={bbox}
              color="red"
              opacity={0.3}
              onClick={() => {
                setSelectedBBoxIndex(i);
                setSelectedAutoTagId(null);
              }}
              isSelected={selectedBBoxIndex === i && selectedAutoTagId === null}
              isDimmed={
                (selectedBBoxIndex !== null && selectedBBoxIndex !== i) ||
                selectedAutoTagId !== null
              }
            />
          ))}

          {showAutoTags &&
            autoTagBBoxes?.map((bbox, i) => (
              <BoundingBoxMesh
                key={`autotag-${i}`}
                bbox={bbox}
                color={new THREE.Color().setHSL((i * 0.618) % 1, 0.8, 0.5)}
                label={annotations?.[i]}
                opacity={0.1}
                onClick={() => {
                  if (annotations?.[i]) {
                    setSelectedAutoTagId(annotations[i]);
                    setSelectedBBoxIndex(null);
                  }
                }}
                isSelected={selectedAutoTagId === annotations?.[i]}
                isDimmed={
                  selectedBBoxIndex !== null ||
                  (selectedAutoTagId !== null &&
                    selectedAutoTagId !== annotations?.[i])
                }
              />
            ))}

          {showOccupancyGrid &&
            occupancyGrid?.map((bbox, i) => (
              <BoundingBoxMesh
                key={`occupancy-${i}`}
                bbox={bbox}
                color="green"
                opacity={0.1}
                isDimmed={selectedBBoxIndex !== null}
              />
            ))}

          {route && <RoutePath route={route} />}
        </Canvas>
      </KeyboardControls>
    </div>
  );
}

function RoutePath({ route }: { route: Route }) {
  if (!route || route.length < 2) return null;
  const points = useMemo(
    () => route.map((pt) => new THREE.Vector3(pt[0], pt[1], pt[2])),
    [route],
  );
  return <Line points={points} color="blue" lineWidth={3} />;
}
