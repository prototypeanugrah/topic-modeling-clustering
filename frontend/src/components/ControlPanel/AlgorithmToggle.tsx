export type Algorithm = "kmeans" | "gmm";

interface AlgorithmToggleProps {
  value: Algorithm;
  onChange: (value: Algorithm) => void;
}

export function AlgorithmToggle({ value, onChange }: AlgorithmToggleProps) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
        Clustering Algorithm
      </label>
      <div className="flex rounded-lg p-1" style={{ background: "var(--bg-tertiary)" }}>
        <button
          onClick={() => onChange("kmeans")}
          className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
            value === "kmeans"
              ? "shadow-sm"
              : "hover:opacity-80"
          }`}
          style={{
            background: value === "kmeans" ? "var(--bg-card)" : "transparent",
            color: value === "kmeans" ? "var(--accent-primary)" : "var(--text-secondary)",
          }}
        >
          K-Means
        </button>
        <button
          onClick={() => onChange("gmm")}
          className={`flex-1 px-3 py-1.5 text-sm font-medium rounded-md transition-all ${
            value === "gmm"
              ? "shadow-sm"
              : "hover:opacity-80"
          }`}
          style={{
            background: value === "gmm" ? "var(--bg-card)" : "transparent",
            color: value === "gmm" ? "var(--accent-primary)" : "var(--text-secondary)",
          }}
        >
          GMM
        </button>
      </div>
    </div>
  );
}
