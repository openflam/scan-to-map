import { useGLTF } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

/**
 * Matches search-server/app.py transform_coordinates / transform_bbox:
 * viewer (x,y,z) = (colmap_y, colmap_z, colmap_x).
 * ScanNet / COLMAP meshes are stored in world space; API already remaps boxes
 * for the viewer, so the mesh must use the same linear map.
 */
function colmapWorldToViewerMatrix(): THREE.Matrix4 {
  const m = new THREE.Matrix4();
  m.set(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
  return m;
}

export default function Model({
  url,
  datasetName,
}: {
  url: string;
  /** When name starts with scannet_, align mesh to the same frame as transformed bboxes. */
  datasetName?: string | null;
}) {
  const { scene } = useGLTF(url);

  const alignMeshToViewer =
    typeof datasetName === "string" && datasetName.startsWith("scannet_");

  const viewerAlignMatrix = useMemo(
    () => (alignMeshToViewer ? colmapWorldToViewerMatrix() : null),
    [alignMeshToViewer],
  );

  useMemo(() => {
    scene.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        if (child.material) {
          const materials = Array.isArray(child.material)
            ? child.material
            : [child.material];

          materials.forEach((mat) => {
            if (
              mat instanceof THREE.MeshStandardMaterial ||
              mat instanceof THREE.MeshPhysicalMaterial
            ) {
              // Tone down metalness to prevent pure black reflection rendering
              mat.metalness = Math.min(mat.metalness, 0.2);
              mat.roughness = Math.max(mat.roughness, 0.8);

              // Enable vertex colors only if no texture map exists and colors are present
              if (
                !mat.map &&
                child.geometry &&
                child.geometry.attributes.color
              ) {
                mat.vertexColors = true;
                mat.color.setHex(0xffffff); // Ensure base color doesn't darken vertex colors
              }

              mat.needsUpdate = true;
            }
          });
        }
      }
    });
  }, [scene]);

  const primitive = <primitive object={scene} />;

  if (viewerAlignMatrix) {
    return (
      <group matrix={viewerAlignMatrix} matrixAutoUpdate={false}>
        {primitive}
      </group>
    );
  }

  return primitive;
}
