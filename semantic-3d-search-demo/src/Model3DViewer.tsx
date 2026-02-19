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
  TransformControls,
} from "@react-three/drei";
import {
  Suspense,
  useMemo,
  useState,
  useEffect,
  useLayoutEffect,
  useRef,
} from "react";
import * as THREE from "three";
import type { BoundingBox, Route } from "./types/global";
import ComponentDetails from "./ComponentDetails";
import type { GizmoMode } from "./ComponentDetails";

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

function BBoxTransformGizmo({
  initialBBox,
  mode,
  onCommit,
}: {
  initialBBox: BoundingBox;
  mode: GizmoMode;
  onCommit: (bbox: BoundingBox) => void;
}) {
  const meshRef = useRef<THREE.Mesh>(null);
  const tcRef = useRef<any>(null);
  // Track live transforms in a ref – no React state updates during drag
  const liveBBoxRef = useRef<BoundingBox>(initialBBox);

  // Position/scale the mesh imperatively before the first Three.js frame so the
  // gizmo spawns at the bbox center. useLayoutEffect fires synchronously after
  // the R3F reconciler commits but before the canvas draws.
  useLayoutEffect(() => {
    const m = meshRef.current;
    if (!m) return;
    const { x_min, y_min, z_min, x_max, y_max, z_max } = initialBBox;
    m.position.set(
      (x_min + x_max) / 2,
      (y_min + y_max) / 2,
      (z_min + z_max) / 2,
    );
    m.scale.set(
      Math.abs(x_max - x_min),
      Math.abs(y_max - y_min),
      Math.abs(z_max - z_min),
    );
    m.updateMatrixWorld(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount

  // Wire raw Three.js events so drag tracking never triggers React re-renders
  useEffect(() => {
    const tc = tcRef.current;
    if (!tc) return;

    const onObjectChange = () => {
      const m = meshRef.current;
      if (!m) return;
      liveBBoxRef.current = {
        x_min: m.position.x - m.scale.x / 2,
        x_max: m.position.x + m.scale.x / 2,
        y_min: m.position.y - m.scale.y / 2,
        y_max: m.position.y + m.scale.y / 2,
        z_min: m.position.z - m.scale.z / 2,
        z_max: m.position.z + m.scale.z / 2,
      };
    };

    const onDraggingChanged = (e: any) => {
      if (!e.value) onCommit(liveBBoxRef.current);
    };

    tc.addEventListener("objectChange", onObjectChange);
    tc.addEventListener("dragging-changed", onDraggingChanged);
    return () => {
      tc.removeEventListener("objectChange", onObjectChange);
      tc.removeEventListener("dragging-changed", onDraggingChanged);
    };
  }, [onCommit]);

  // Render the mesh and TransformControls as siblings.
  // Using the `object` prop (instead of children wrapping) makes TransformControls
  // attach() directly to the mesh, so the gizmo lives at the mesh's world position
  // and moves with it. The children-wrapper approach puts kids inside an internal
  // group at origin, causing the gizmo to appear at (0,0,0).
  return (
    <>
      {/* No position/scale JSX props – set imperatively above so re-renders
          from onCommit → setEditedBBox never reset the Three.js transform */}
      <mesh ref={meshRef}>
        <boxGeometry args={[1, 1, 1]} />
        <meshStandardMaterial
          color="#3b82f6"
          transparent
          opacity={0.35}
          depthWrite={false}
          side={THREE.DoubleSide}
        />
        <Edges color="#1d4ed8" />
      </mesh>
      <TransformControls
        ref={tcRef}
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        object={meshRef as any}
        mode={mode}
      />
    </>
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
  const [editedBBox, setEditedBBox] = useState<BoundingBox | null>(null);
  const [gizmoMode, setGizmoMode] = useState<GizmoMode>("translate");
  // savedBBoxMap stores saved positions keyed by component ID so the box
  // stays in its new position even after the panel is dismissed/deselected.
  const [savedBBoxMap, setSavedBBoxMap] = useState<Map<string, BoundingBox>>(
    new Map(),
  );
  const [saveWarning, setSaveWarning] = useState<string | null>(null);
  // Disable OrbitControls as soon as the gizmo is visible so it never
  // interferes with pointer events intended for TransformControls.
  const orbitEnabled = !(isEditing && editedBBox !== null);

  // Derive the bbox of whichever item is currently selected, preferring any
  // previously saved position so the gizmo starts from the right place if
  // the user opens edit mode again after a save.
  const selectedBBoxData: BoundingBox | null = useMemo(() => {
    let rawBBox: BoundingBox | null = null;
    let key: string | null = null;
    if (selectedBBoxIndex !== null && boundingBox) {
      rawBBox = boundingBox[selectedBBoxIndex] ?? null;
      key = componentIds?.[selectedBBoxIndex] ?? null;
    } else if (selectedAutoTagId !== null && annotations && autoTagBBoxes) {
      const idx = annotations.indexOf(selectedAutoTagId);
      rawBBox = idx >= 0 ? (autoTagBBoxes[idx] ?? null) : null;
      key = selectedAutoTagId;
    }
    if (key && savedBBoxMap.has(key)) return savedBBoxMap.get(key)!;
    return rawBBox;
  }, [
    selectedBBoxIndex,
    selectedAutoTagId,
    boundingBox,
    annotations,
    autoTagBBoxes,
    componentIds,
    savedBBoxMap,
  ]);

  // Initialise editedBBox when entering edit mode
  useEffect(() => {
    if (isEditing) {
      setEditedBBox(selectedBBoxData);
    } else {
      setEditedBBox(null);
    }
  }, [isEditing, selectedBBoxData]);

  // Clear the save warning whenever the selection changes
  useEffect(() => {
    setSaveWarning(null);
  }, [selectedBBoxIndex, selectedAutoTagId]);

  const handleSave = async () => {
    // Capture before setIsEditing(false) clears editedBBox via useEffect
    const bboxToSave = editedBBox;
    const captionToSave = editedCaption;
    // Persist the new position immediately so the box doesn't snap back when
    // isEditing → false unmounts the gizmo and restores the BoundingBoxMesh.
    if (bboxToSave && currentComponentId) {
      setSavedBBoxMap((prev) =>
        new Map(prev).set(currentComponentId, bboxToSave),
      );
    }
    setIsEditing(false);
    if (!currentComponentId) return;
    try {
      const { updateComponent } = await import("./query");
      const updates: {
        caption: string;
        bbox?: { min: number[]; max: number[] };
      } = {
        caption: captionToSave,
      };
      if (bboxToSave) {
        // Invert the axis swap applied in App.tsx when loading from the server:
        //   viewer.x = server[1], viewer.y = server[2], viewer.z = server[0]
        // → server[0] = viewer.z, server[1] = viewer.x, server[2] = viewer.y
        updates.bbox = {
          min: [bboxToSave.z_min, bboxToSave.x_min, bboxToSave.y_min],
          max: [bboxToSave.z_max, bboxToSave.x_max, bboxToSave.y_max],
        };
      }
      await updateComponent(currentComponentId, updates);
      setSaveWarning(null);
    } catch (error) {
      console.error("Error saving component:", error);
      setSaveWarning("Save failed — changes may not have been persisted.");
    }
  };

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
            onSave={handleSave}
            onDelete={
              currentComponentId
                ? async () => {
                    try {
                      const { deleteComponent } = await import("./query");
                      await deleteComponent(currentComponentId);
                      setSelectedBBoxIndex(null);
                      setSelectedAutoTagId(null);
                    } catch (error) {
                      console.error("Error deleting component:", error);
                    }
                  }
                : undefined
            }
            saveWarning={saveWarning}
            componentId={currentComponentId}
            imageBase64={componentImage}
            isLoading={isLoadingComponent}
            editedBBox={editedBBox}
            gizmoMode={gizmoMode}
            onGizmoModeChange={setGizmoMode}
          />
        )}

        {/* --- 3D SCENE --- */}
        <Canvas
          camera={{ position: [5, 5, 5] }}
          /* Removed onPointerMissed to prevent deselection on background click */
        >
          <ambientLight intensity={2.0} />
          <pointLight position={[10, 10, 10]} intensity={2.0} />
          <OrbitControls makeDefault enabled={orbitEnabled} />
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

          {boundingBox?.map((bbox, i) => {
            const isThisSelected =
              selectedBBoxIndex === i && selectedAutoTagId === null;
            if (isThisSelected && isEditing) return null;
            const savedId = componentIds?.[i] ?? null;
            const displayBBox =
              savedId && savedBBoxMap.has(savedId)
                ? savedBBoxMap.get(savedId)!
                : bbox;
            return (
              <BoundingBoxMesh
                key={`bbox-${i}`}
                bbox={displayBBox}
                color="red"
                opacity={0.3}
                onClick={() => {
                  setSelectedBBoxIndex(i);
                  setSelectedAutoTagId(null);
                }}
                isSelected={isThisSelected}
                isDimmed={
                  (selectedBBoxIndex !== null && selectedBBoxIndex !== i) ||
                  selectedAutoTagId !== null
                }
              />
            );
          })}

          {showAutoTags &&
            autoTagBBoxes?.map((bbox, i) => {
              const isThisSelected = selectedAutoTagId === annotations?.[i];
              if (isThisSelected && isEditing) return null;
              const annotId = annotations?.[i] ?? null;
              const displayBBox =
                annotId && savedBBoxMap.has(annotId)
                  ? savedBBoxMap.get(annotId)!
                  : bbox;
              return (
                <BoundingBoxMesh
                  key={`autotag-${i}`}
                  bbox={displayBBox}
                  color={new THREE.Color().setHSL((i * 0.618) % 1, 0.8, 0.5)}
                  label={annotations?.[i]}
                  opacity={0.1}
                  onClick={() => {
                    if (annotations?.[i]) {
                      setSelectedAutoTagId(annotations[i]);
                      setSelectedBBoxIndex(null);
                    }
                  }}
                  isSelected={isThisSelected}
                  isDimmed={
                    selectedBBoxIndex !== null ||
                    (selectedAutoTagId !== null &&
                      selectedAutoTagId !== annotations?.[i])
                  }
                />
              );
            })}

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

          {isEditing && editedBBox && (
            <BBoxTransformGizmo
              key="bbox-gizmo"
              initialBBox={editedBBox}
              mode={gizmoMode}
              onCommit={setEditedBBox}
            />
          )}
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
