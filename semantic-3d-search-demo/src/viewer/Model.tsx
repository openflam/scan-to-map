import { useGLTF } from "@react-three/drei";

export default function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  return <primitive object={scene} />;
}
