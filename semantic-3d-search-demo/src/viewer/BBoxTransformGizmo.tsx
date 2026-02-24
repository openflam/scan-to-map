import { useEffect, useLayoutEffect, useRef } from "react";
import { TransformControls, Edges } from "@react-three/drei";
import * as THREE from "three";
import type { BoundingBox } from "../types/global";
import type { GizmoMode } from "../ComponentDetails";

interface BBoxTransformGizmoProps {
  initialBBox: BoundingBox;
  mode: GizmoMode;
  onCommit: (bbox: BoundingBox) => void;
}

export default function BBoxTransformGizmo({
  initialBBox,
  mode,
  onCommit,
}: BBoxTransformGizmoProps) {
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
