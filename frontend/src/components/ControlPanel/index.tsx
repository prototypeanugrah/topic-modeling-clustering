import { TopicSlider } from "./TopicSlider";
import { ClusterSlider } from "./ClusterSlider";

interface ControlPanelProps {
  nTopics: number;
  nClusters: number;
  onTopicsChange: (value: number) => void;
  onClustersChange: (value: number) => void;
  optimalTopics?: number;
  optimalClusters?: number;
  minTopics?: number;
  maxTopics?: number;
}

export function ControlPanel({
  nTopics,
  nClusters,
  onTopicsChange,
  onClustersChange,
  optimalTopics,
  optimalClusters,
  minTopics = 2,
  maxTopics = 20,
}: ControlPanelProps) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
      <div className="flex items-center gap-2">
        <svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
        </svg>
        <h2 className="text-lg font-semibold text-gray-800">Controls</h2>
      </div>

      <TopicSlider
        value={nTopics}
        onChange={onTopicsChange}
        min={minTopics}
        max={maxTopics}
        optimalValue={optimalTopics}
      />

      <ClusterSlider
        value={nClusters}
        onChange={onClustersChange}
        optimalValue={optimalClusters}
      />
    </div>
  );
}
