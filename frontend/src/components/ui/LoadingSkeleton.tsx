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
    <div className={`bg-white rounded-xl shadow-lg p-6 ${height} flex flex-col`}>
      {title && (
        <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
      )}
      <div className="flex-1 flex flex-col items-center justify-center">
        <div className="relative">
          <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-200"></div>
          <div className="animate-spin rounded-full h-10 w-10 border-4 border-indigo-600 border-t-transparent absolute top-0 left-0"></div>
        </div>
        <span className="text-gray-500 text-sm mt-4">{message}</span>
      </div>
    </div>
  );
}

export function LoadingCard({
  height = "h-64",
  className = ""
}: { height?: string; className?: string }) {
  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 ${height} ${className} animate-pulse`}>
      <div className="h-4 bg-gray-200 rounded w-1/3 mb-4"></div>
      <div className="flex-1 flex flex-col justify-center space-y-3">
        <div className="h-3 bg-gray-200 rounded w-full"></div>
        <div className="h-3 bg-gray-200 rounded w-5/6"></div>
        <div className="h-3 bg-gray-200 rounded w-4/6"></div>
      </div>
    </div>
  );
}
