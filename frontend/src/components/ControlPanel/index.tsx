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
  minClusters?: number;
  maxClusters?: number;
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
  minClusters = 2,
  maxClusters = 15,
}: ControlPanelProps) {
  return (
    <div
      className="terminal-panel lg:shadow-none lg:border-0 lg:rounded-none"
      style={{ background: 'var(--bg-card)' }}
    >
      <div className="terminal-panel-header lg:hidden">
        <svg className="w-4 h-4" style={{ color: 'var(--accent-primary)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
        </svg>
        Controls
      </div>

      <div className="terminal-panel-content lg:p-0 space-y-6">
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
          min={minClusters}
          max={maxClusters}
          optimalValue={optimalClusters}
        />
      </div>
    </div>
  );
}
