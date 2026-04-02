import type { BoundingBox } from "../types/global";
import { styles } from "../componentDetailsStyles";

export type GizmoMode = "translate" | "scale";

export interface ComponentDetailsProps {
  editedCaption: string;
  setEditedCaption: (value: string) => void;
  isEditing: boolean;
  setIsEditing: (value: boolean) => void;
  onDismiss: () => void;
  onSave: () => void;
  saveWarning?: string | null;
  componentId: string | null;
  imageBase64?: string | null;
  isLoading?: boolean;
  editedBBox?: BoundingBox | null;
  gizmoMode?: GizmoMode;
  onGizmoModeChange?: (mode: GizmoMode) => void;
  onDelete?: () => void;
}

export default function BenchmarkComponentDetails({
  editedCaption,
  onDismiss,
  componentId,
  imageBase64,
  isLoading,
}: ComponentDetailsProps) {
  return (
    <div style={styles.panel}>
      <h3 style={styles.sectionLabel}>Component {componentId}</h3>

      {isLoading ? (
        <div style={styles.loadingContainer}>
          <p style={styles.loadingText}>Loading...</p>
        </div>
      ) : (
        <>
          {imageBase64 && (
            <div style={styles.imageWrapper}>
              <img
                src={`data:image/jpeg;base64,${imageBase64}`}
                alt="Component view"
                style={styles.image}
              />
            </div>
          )}

          <div style={styles.captionBox}>
            <p style={styles.captionText}>
              {editedCaption || "No caption available."}
            </p>
          </div>
        </>
      )}

      <button onClick={onDismiss} style={{ ...styles.dismissButton, marginTop: '20px' }}>
        Dismiss (Esc)
      </button>
    </div>
  );
}
