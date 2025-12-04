import { useState, useEffect } from "react";
import { useTheme } from "../../hooks/useTheme";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface PyLDAvisViewerProps {
  nTopics: number;
}

export function PyLDAvisViewer({ nTopics }: PyLDAvisViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  const { isDark } = useTheme();

  const theme = isDark ? "dark" : "light";
  const iframeSrc = `${API_BASE}/api/topics/${nTopics}/pyldavis?theme=${theme}`;

  // Reset loading state when nTopics or theme changes
  useEffect(() => {
    setLoading(true);
    setError(false);
  }, [nTopics, theme]);

  if (error) {
    return (
      <div className="terminal-panel">
        <div className="terminal-panel-header">
          <span className="status-dot status-dot--warning"></span>
          pyLDAvis Visualization
        </div>
        <div className="terminal-panel-content">
          <div
            className="h-[880px] flex flex-col items-center justify-center rounded"
            style={{ background: 'var(--bg-secondary)', border: '1px dashed var(--border-color)' }}
          >
            <svg className="w-12 h-12 mb-3" style={{ color: 'var(--text-muted)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
            <p className="font-mono text-sm" style={{ color: 'var(--text-secondary)' }}>pyLDAvis not available</p>
            <p className="font-mono text-xs mt-1" style={{ color: 'var(--text-muted)' }}>Run precompute.py to generate visualizations</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="terminal-panel">
      <div className="terminal-panel-header">
        pyLDAvis Visualization
      </div>
      <div className="terminal-panel-content relative">
        {loading && (
          <div
            className="absolute inset-0 rounded flex flex-col items-center justify-center z-10"
            style={{ background: 'var(--bg-card)' }}
          >
            <div className="terminal-loading">Loading interactive visualization</div>
            <div className="loading-bar w-64 mt-4">
              <div className="loading-bar-progress"></div>
            </div>
          </div>
        )}
        <iframe
          key={`${nTopics}-${theme}`}
          src={iframeSrc}
          className="w-full h-[880px] border-0 rounded"
          style={{ background: isDark ? '#21262d' : '#ffffff' }}
          title="pyLDAvis visualization"
          onLoad={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setError(true);
          }}
        />
      </div>
      <div
        className="px-4 py-2 text-center font-mono text-xs"
        style={{ color: 'var(--text-muted)', borderTop: '1px solid var(--border-color)' }}
      >
        Interactive visualization: Hover over topics to see word distributions. Click topics to select.
      </div>
    </div>
  );
}
