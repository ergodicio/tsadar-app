/** Loading, empty and error states. Kept explicit rather than rendering an
 *  ambiguous blank table. */

interface ErrorStateProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="state state--error" role="alert">
      <p className="state__title">Could not load runs</p>
      <p className="state__detail">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="button">
          Try again
        </button>
      )}
    </div>
  );
}

export function LoadingState() {
  return (
    <div className="state" role="status" aria-live="polite">
      <p className="state__title">Loading runs…</p>
    </div>
  );
}

interface EmptyStateProps {
  filtered: boolean;
  onClear?: () => void;
}

export function EmptyState({ filtered, onClear }: EmptyStateProps) {
  // "No runs match" and "there are no runs" are different problems and lead to
  // different next actions, so they get different copy.
  return (
    <div className="state" role="status">
      <p className="state__title">{filtered ? "No runs match these filters" : "No runs yet"}</p>
      <p className="state__detail">
        {filtered
          ? "Try widening the shot number, status or experiment."
          : "Runs appear here once a fit has been logged to MLflow."}
      </p>
      {filtered && onClear && (
        <button type="button" onClick={onClear} className="button">
          Clear filters
        </button>
      )}
    </div>
  );
}
