import { useTheme } from '../../hooks/useTheme';

export function ThemeToggle() {
  const { isDark, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="flex items-center gap-2 px-3 py-2 font-mono text-xs uppercase tracking-wider transition-all duration-150 border rounded"
      style={{
        background: 'var(--bg-secondary)',
        borderColor: 'var(--border-color)',
        color: 'var(--text-secondary)',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--accent-primary)';
        e.currentTarget.style.borderColor = 'var(--accent-primary)';
        e.currentTarget.style.color = 'white';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'var(--bg-secondary)';
        e.currentTarget.style.borderColor = 'var(--border-color)';
        e.currentTarget.style.color = 'var(--text-secondary)';
      }}
      aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
    >
      {/* Sun icon for light mode */}
      <svg
        className={`w-4 h-4 transition-opacity duration-150 ${isDark ? 'opacity-40' : 'opacity-100'}`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <circle cx="12" cy="12" r="5" />
        <line x1="12" y1="1" x2="12" y2="3" />
        <line x1="12" y1="21" x2="12" y2="23" />
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
        <line x1="1" y1="12" x2="3" y2="12" />
        <line x1="21" y1="12" x2="23" y2="12" />
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
      </svg>

      {/* Toggle indicator */}
      <div
        className="relative w-8 h-4 rounded-sm transition-colors duration-150"
        style={{
          background: isDark ? 'var(--accent-primary)' : 'var(--border-color)',
        }}
      >
        <div
          className="absolute top-0.5 w-3 h-3 rounded-sm transition-transform duration-150"
          style={{
            background: 'var(--bg-card)',
            transform: isDark ? 'translateX(18px)' : 'translateX(2px)',
          }}
        />
      </div>

      {/* Moon icon for dark mode */}
      <svg
        className={`w-4 h-4 transition-opacity duration-150 ${isDark ? 'opacity-100' : 'opacity-40'}`}
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
      >
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
      </svg>
    </button>
  );
}
