import type { BoundingBox } from "./types/global";

export type GizmoMode = "translate" | "scale";

interface ComponentDetailsProps {
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
}

export default function ComponentDetails({
  editedCaption,
  setEditedCaption,
  isEditing,
  setIsEditing,
  onDismiss,
  onSave,
  saveWarning,
  componentId: _componentId,
  imageBase64,
  isLoading,
  editedBBox: _editedBBox,
  gizmoMode,
  onGizmoModeChange,
}: ComponentDetailsProps) {
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
        {isEditing && onGizmoModeChange && (
          <div
            style={{
              display: "flex",
              gap: "6px",
              marginBottom: "4px",
              width: "100%",
            }}
          >
            <span
              style={{
                fontSize: "11px",
                color: "#888",
                alignSelf: "center",
                marginRight: "4px",
              }}
            >
              Gizmo:
            </span>
            {(["translate", "scale"] as GizmoMode[]).map((m) => (
              <button
                key={m}
                onClick={() => onGizmoModeChange(m)}
                style={{
                  flex: 1,
                  padding: "6px",
                  cursor: "pointer",
                  backgroundColor: gizmoMode === m ? "#3b82f6" : "#f3f4f6",
                  color: gizmoMode === m ? "white" : "#374151",
                  border: `1px solid ${gizmoMode === m ? "#3b82f6" : "#d1d5db"}`,
                  borderRadius: "6px",
                  fontSize: "12px",
                  fontWeight: "600",
                  textTransform: "capitalize",
                  transition: "all 0.2s",
                }}
              >
                {m}
              </button>
            ))}
          </div>
        )}
      </div>

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
          onClick={onSave}
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
      {saveWarning && (
        <div
          style={{
            padding: "8px 12px",
            borderRadius: "6px",
            backgroundColor: "#fef2f2",
            border: "1px solid #fecaca",
            color: "#b91c1c",
            fontSize: "12px",
          }}
        >
          {saveWarning}
        </div>
      )}
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
