import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface PyLDAvisViewerProps {
  nTopics: number;
}

export function PyLDAvisViewer({ nTopics }: PyLDAvisViewerProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const iframeSrc = `${API_BASE}/api/topics/${nTopics}/pyldavis`;

  if (error) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Topic-Word Distribution (pyLDAvis)
        </h3>
        <div className="h-[800px] flex flex-col items-center justify-center bg-gray-50 rounded-lg border-2 border-dashed border-gray-200">
          <svg className="w-12 h-12 text-gray-400 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
          <p className="text-gray-500 text-sm mb-2">pyLDAvis not available</p>
          <p className="text-gray-400 text-xs">Run precompute.py to generate visualizations</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Topic-Word Distribution (pyLDAvis)
      </h3>
      <div className="relative">
        {loading && (
          <div className="absolute inset-0 bg-white rounded-lg flex flex-col items-center justify-center z-10">
            <div className="relative">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-200"></div>
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent absolute top-0 left-0"></div>
            </div>
            <span className="text-gray-500 text-sm mt-4">Loading interactive visualization...</span>
          </div>
        )}
        <iframe
          key={nTopics}
          src={iframeSrc}
          className="w-full h-[800px] border-0 rounded-lg bg-white"
          title="pyLDAvis visualization"
          onLoad={() => setLoading(false)}
          onError={() => {
            setLoading(false);
            setError(true);
          }}
        />
      </div>
      <p className="text-xs text-gray-400 mt-3 text-center">
        Interactive visualization: Hover over topics to see word distributions. Click topics to select.
      </p>
    </div>
  );
}
