interface ClusterSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  optimalValue?: number;
}

export function ClusterSlider({
  value,
  onChange,
  min = 2,
  max = 15,
  optimalValue,
}: ClusterSliderProps) {
  const decrement = () => {
    if (value > min) onChange(value - 1);
  };

  const increment = () => {
    if (value < max) onChange(value + 1);
  };

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <label className="data-label">
          Clusters
        </label>
        <span
          className="data-value text-lg"
          style={{ color: 'var(--accent-primary)' }}
        >
          {value}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <button
          onClick={decrement}
          disabled={value <= min}
          className="btn-circle"
          aria-label="Decrease clusters"
        >
          &lt;
        </button>

        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(parseInt(e.target.value))}
          className="flex-1 cursor-pointer"
        />

        <button
          onClick={increment}
          disabled={value >= max}
          className="btn-circle"
          aria-label="Increase clusters"
        >
          &gt;
        </button>
      </div>

      <div className="flex justify-between items-center font-mono text-xs" style={{ color: 'var(--text-secondary)' }}>
        <span>{min}</span>
        {optimalValue && (
          <span
            className="px-2 py-0.5 rounded"
            style={{ background: 'var(--accent-secondary)', color: 'white' }}
          >
            optimal: {optimalValue}
          </span>
        )}
        <span>{max}</span>
      </div>
    </div>
  );
}
