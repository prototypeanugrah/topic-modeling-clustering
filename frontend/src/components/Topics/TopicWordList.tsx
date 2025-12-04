import type { TopicWordsResponse } from "../../types/api";

interface TopicWordListProps {
  data: TopicWordsResponse | null;
  loading: boolean;
  isValidating?: boolean;
}

export function TopicWordList({ data, loading, isValidating }: TopicWordListProps) {
  // Show full skeleton only on initial load (no cached data)
  if (loading && !data) {
    return (
      <div className="terminal-panel">
        <div className="terminal-panel-header">
          Topic Words
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-8">
          <div className="terminal-loading">Extracting topic words</div>
          <div className="loading-bar w-48 mt-4">
            <div className="loading-bar-progress"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="terminal-panel">
        <div className="terminal-panel-header">
          <span className="status-dot status-dot--error"></span>
          Topic Words
        </div>
        <div className="terminal-panel-content flex flex-col items-center justify-center py-8">
          <svg className="w-10 h-10 mb-2" style={{ color: 'var(--status-error)' }} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="font-mono text-sm" style={{ color: 'var(--status-error)' }}>Failed to load topic words</span>
        </div>
      </div>
    );
  }

  // Show subtle indicator during background revalidation
  const showValidatingIndicator = isValidating && data;

  // Determine max words to show (use first topic's length, max 6)
  const maxWords = Math.min(data.topics[0]?.length || 6, 6);

  return (
    <div className={`terminal-panel relative transition-opacity ${showValidatingIndicator ? 'opacity-80' : ''}`}>
      {showValidatingIndicator && (
        <div className="absolute top-3 right-3 flex items-center gap-2 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          <div className="status-dot status-dot--active" style={{ width: 6, height: 6 }} />
          <span>Updating...</span>
        </div>
      )}
      <div className="terminal-panel-header">
        Topic Words
        <span className="ml-2 px-2 py-0.5 rounded text-xs" style={{ background: 'var(--accent-primary)', color: 'white' }}>
          {data.n_topics} topics
        </span>
      </div>
      <div className="terminal-panel-content p-0">
        <div className="overflow-x-auto custom-scrollbar">
          <table className="terminal-table">
            <thead>
              <tr>
                <th className="whitespace-nowrap">Topic</th>
                {Array.from({ length: maxWords }, (_, i) => (
                  <th key={i} className="whitespace-nowrap">
                    Word {i + 1}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.topics.map((topic, idx) => (
                <tr key={idx}>
                  <td className="whitespace-nowrap">
                    <span className="inline-flex items-center gap-2">
                      <span
                        className="inline-flex items-center justify-center w-6 h-6 rounded text-xs font-semibold"
                        style={{ background: 'var(--accent-primary)', color: 'white' }}
                      >
                        {idx + 1}
                      </span>
                      <span className="font-medium" style={{ color: 'var(--text-primary)' }}>
                        Topic {idx + 1}
                      </span>
                    </span>
                  </td>
                  {topic.slice(0, maxWords).map((word, wordIdx) => (
                    <td key={wordIdx} className="whitespace-nowrap">
                      <span style={{ color: 'var(--text-primary)' }}>{word.word}</span>
                      <span className="text-xs ml-1" style={{ color: 'var(--text-muted)' }}>
                        ({(word.probability * 100).toFixed(1)}%)
                      </span>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
