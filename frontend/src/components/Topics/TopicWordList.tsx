import type { TopicWordsResponse } from "../../types/api";

interface TopicWordListProps {
  data: TopicWordsResponse | null;
  loading: boolean;
}

export function TopicWordList({ data, loading }: TopicWordListProps) {
  if (loading) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Top Words per Topic
        </h3>
        <div className="flex flex-col items-center justify-center py-8">
          <div className="relative">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-green-200"></div>
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-green-600 border-t-transparent absolute top-0 left-0"></div>
          </div>
          <span className="text-gray-500 text-sm mt-3">Extracting topic words...</span>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">
          Top Words per Topic
        </h3>
        <div className="flex flex-col items-center justify-center py-8">
          <svg className="w-10 h-10 text-red-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
          <span className="text-red-500 text-sm">Failed to load topic words</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 h-full">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">
        Top Words per Topic
        <span className="text-sm font-normal text-gray-500 ml-2">({data.n_topics} topics)</span>
      </h3>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
        {data.topics.map((topic, idx) => (
          <div
            key={idx}
            className="bg-gradient-to-r from-gray-50 to-white rounded-lg p-3 border border-gray-100 hover:border-indigo-200 transition-colors"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="inline-flex items-center justify-center w-6 h-6 rounded-full bg-indigo-100 text-indigo-700 text-xs font-semibold">
                {idx + 1}
              </span>
              <span className="font-medium text-gray-700 text-sm">Topic {idx + 1}</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {topic.slice(0, 6).map((word, wordIdx) => (
                <span
                  key={wordIdx}
                  className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-700 hover:bg-indigo-100 hover:text-indigo-700 transition-colors cursor-default"
                  title={`${(word.probability * 100).toFixed(1)}%`}
                >
                  {word.word}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
