import { Billboard, Edges, Text } from "@react-three/drei";
import * as THREE from "three";
import type { BoundingBox } from "../types/global";

interface BoundingBoxMeshProps {
  bbox: BoundingBox;
  color: string | THREE.Color;
  label?: string;
  opacity?: number;
  onClick?: (e: any) => void;
  isSelected?: boolean;
  isDimmed?: boolean;
}

export default function BoundingBoxMesh({
  bbox,
  color,
  label,
  opacity = 0.1,
  onClick,
  isSelected = false,
  isDimmed = false,
}: BoundingBoxMeshProps) {
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
