import type { CovarianceType } from "../../types/api";

interface CovarianceSelectorProps {
  value: CovarianceType;
  onChange: (value: CovarianceType) => void;
}

const COVARIANCE_OPTIONS: { value: CovarianceType; label: string; description: string }[] = [
  {
    value: "full",
    label: "Full",
    description: "Each cluster has its own general covariance matrix (most flexible)",
  },
  {
    value: "diag",
    label: "Diagonal",
    description: "Axis-aligned ellipsoids (balanced)",
  },
  {
    value: "spherical",
    label: "Spherical",
    description: "Spherical clusters like K-Means (fastest)",
  },
];

export function CovarianceSelector({ value, onChange }: CovarianceSelectorProps) {
  return (
    <div className="space-y-2">
      <label className="text-xs font-medium" style={{ color: "var(--text-secondary)" }}>
        Covariance Type
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as CovarianceType)}
        className="w-full px-3 py-2 text-sm rounded-lg border transition-colors cursor-pointer"
        style={{
          background: "var(--bg-tertiary)",
          borderColor: "var(--border-primary)",
          color: "var(--text-primary)",
        }}
        title={COVARIANCE_OPTIONS.find((opt) => opt.value === value)?.description}
      >
        {COVARIANCE_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <p className="text-xs" style={{ color: "var(--text-muted)" }}>
        {COVARIANCE_OPTIONS.find((opt) => opt.value === value)?.description}
      </p>
    </div>
  );
}
