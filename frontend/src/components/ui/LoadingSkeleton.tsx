interface LoadingSkeletonProps {
  height?: string;
  title?: string;
  message?: string;
}

export function LoadingSkeleton({
  height = "h-64",
  title,
  message = "Calculating..."
}: LoadingSkeletonProps) {
  return (
    <div className={`terminal-panel ${height} flex flex-col`}>
      {title && (
        <div className="terminal-panel-header">
          {title}
        </div>
      )}
      <div className="terminal-panel-content flex-1 flex flex-col items-center justify-center">
        <div className="terminal-loading">{message}</div>
        <div className="loading-bar w-48 mt-4">
          <div className="loading-bar-progress"></div>
        </div>
      </div>
    </div>
  );
}

export function LoadingCard({
  height = "h-64",
  className = ""
}: { height?: string; className?: string }) {
  return (
    <div
      className={`terminal-panel ${height} ${className}`}
      style={{ background: 'var(--bg-card)' }}
    >
      <div className="terminal-panel-header">
        <div className="h-3 rounded w-1/3 animate-pulse" style={{ background: 'var(--border-color)' }}></div>
      </div>
      <div className="terminal-panel-content flex-1 flex flex-col justify-center space-y-3">
        <div className="h-2 rounded w-full animate-pulse" style={{ background: 'var(--border-color)' }}></div>
        <div className="h-2 rounded w-5/6 animate-pulse" style={{ background: 'var(--border-color)' }}></div>
        <div className="h-2 rounded w-4/6 animate-pulse" style={{ background: 'var(--border-color)' }}></div>
      </div>
    </div>
  );
}
