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
import { Suspense, useMemo, useState } from "react";
import * as THREE from "three";
import type { BoundingBox, Route } from "./types/global";

interface Model3DViewerProps {
  source: string; // Restricting to string for useGLTF for now, as App.tsx passes string
  boundingBox?: BoundingBox[];
  captions?: string[];
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
  caption,
  showCaption,
}: {
  bbox: BoundingBox;
  color: string | THREE.Color;
  label?: string;
  opacity?: number;
  onClick?: () => void;
  caption?: string;
  showCaption?: boolean;
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

  return (
    <group position={position}>
      <mesh onClick={onClick}>
        <boxGeometry args={[width, height, depth]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={opacity}
          depthWrite={false} // Similar to backFaceCulling=false often implies transparency handling
          side={THREE.DoubleSide}
        />
        <Edges color={new THREE.Color(color).multiplyScalar(0.45)} />
      </mesh>
      {label && (
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
      {showCaption && caption && (
        <Billboard position={[0, height / 2 + 0.2, 0]}>
          <group>
            {/* White background board - dynamically sized to fit text */}
            <mesh position={[0, 0, -0.01]} renderOrder={999}>
              <planeGeometry
                args={[
                  Math.min(Math.max(caption.length * 0.09, 1), 4) + 0.4,
                  0.35 + Math.ceil(caption.length / 35) * 0.2,
                ]}
              />
              <meshBasicMaterial
                color="white"
                side={THREE.DoubleSide}
                depthTest={false}
                transparent={true}
                opacity={0.95}
              />
            </mesh>
            {/* Black text on white background */}
            <Text
              fontSize={0.15}
              color="black"
              anchorX="center"
              anchorY="middle"
              maxWidth={3.6}
              renderOrder={1000}
              material-depthTest={false}
            >
              {caption}
            </Text>
          </group>
        </Billboard>
      )}
    </group>
  );
}

function RoutePath({ route }: { route: Route }) {
  if (!route || route.length < 2) return null;

  const points = useMemo(() => {
    return route.map((pt) => new THREE.Vector3(pt[0], pt[1], pt[2]));
  }, [route]);

  return <Line points={points} color="blue" lineWidth={3} />;
}

function CameraController() {
  const [, get] = useKeyboardControls();
  const { camera } = useThree();

  useFrame((state, delta) => {
    // Prevent movement if the user is typing in a form element
    const activeElement = document.activeElement;
    if (
      activeElement &&
      (activeElement.tagName === "INPUT" ||
        activeElement.tagName === "TEXTAREA" ||
        activeElement.tagName === "SELECT")
    ) {
      return;
    }

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

    // Update OrbitControls target to maintain relative position/rotation pivot
    const controls = state.controls as any;
    if (controls) {
      controls.target.add(direction);
    }
  });

  return null;
}

export default function Model3DViewer({
  source,
  boundingBox,
  captions,
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

  const map = useMemo(
    () => [
      { name: "forward", keys: ["ArrowUp", "KeyW"] },
      { name: "backward", keys: ["ArrowDown", "KeyS"] },
      { name: "left", keys: ["ArrowLeft", "KeyA"] },
      { name: "right", keys: ["ArrowRight", "KeyD"] },
    ],
    [],
  );

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <KeyboardControls map={map}>
        <Canvas camera={{ position: [5, 5, 5] }}>
          <ambientLight intensity={2.0} />
          <pointLight position={[10, 10, 10]} intensity={2.0} />
          <OrbitControls makeDefault />
          <CameraController />

          <Suspense fallback={null}>
            <Model url={source} />
          </Suspense>

          {/* User Search Bounding Boxes */}
          {boundingBox?.map((bbox, i) => (
            <BoundingBoxMesh
              key={`bbox-${i}`}
              bbox={bbox}
              color="red"
              opacity={0.3}
              onClick={() =>
                setSelectedBBoxIndex(selectedBBoxIndex === i ? null : i)
              }
              caption={captions?.[i]}
              showCaption={selectedBBoxIndex === i}
            />
          ))}

          {/* Auto Tag Bounding Boxes */}
          {showAutoTags &&
            autoTagBBoxes?.map((bbox, i) => (
              <BoundingBoxMesh
                key={`autotag-${i}`}
                bbox={bbox}
                // Use deterministic color based on index to avoid flickering on re-renders
                color={new THREE.Color().setHSL(
                  (i * 0.618033988749895) % 1,
                  0.8,
                  0.5,
                )}
                label={annotations?.[i]}
                opacity={0.1}
              />
            ))}

          {/* Occupancy Grid */}
          {showOccupancyGrid &&
            occupancyGrid?.map((bbox, i) => (
              <BoundingBoxMesh
                key={`occupancy-${i}`}
                bbox={bbox}
                color="green"
                opacity={0.1}
              />
            ))}

          {/* Route */}
          {route && <RoutePath route={route} />}
        </Canvas>
      </KeyboardControls>
    </div>
  );
}
