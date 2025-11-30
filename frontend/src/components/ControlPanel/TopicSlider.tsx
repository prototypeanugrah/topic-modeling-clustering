interface TopicSliderProps {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  optimalValue?: number;
}

export function TopicSlider({
  value,
  onChange,
  min = 2,
  max = 20,
  optimalValue,
}: TopicSliderProps) {
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <label className="text-sm font-medium text-gray-700">
          Number of Topics
        </label>
        <span className="text-sm font-bold text-blue-600">{value}</span>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
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
