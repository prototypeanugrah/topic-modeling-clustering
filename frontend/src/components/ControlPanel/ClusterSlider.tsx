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
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-gray-700">
          Number of Clusters
        </label>
        <span className="text-sm font-bold text-purple-600">{value}</span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
      />

      <div className="flex justify-between text-xs text-gray-500">
        <span>{min}</span>
        {optimalValue && (
          <span className="text-green-600 font-medium">
            Optimal: {optimalValue}
          </span>
        )}
        <span>{max}</span>
      </div>
    </div>
  );
}
