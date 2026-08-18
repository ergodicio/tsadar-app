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
  // different next actions, so they get different copy. Both mention the Thomson
  // scope: an empty table is otherwise indistinguishable from a broken one, and
  // "where did my run go" has a different answer here than in the MLflow UI.
  return (
    <div className="state" role="status">
      <p className="state__title">
        {filtered ? "No Thomson runs match these filters" : "No Thomson runs yet"}
      </p>
      <p className="state__detail">
        {filtered
          ? "Try widening the shot number, status or experiment. This browser only covers Thomson scattering experiments, so runs from other projects never appear here."
          : "Runs appear here once a Thomson fit has been logged to MLflow."}
      </p>
      {filtered && onClear && (
        <button type="button" onClick={onClear} className="button">
          Clear filters
        </button>
      )}
    </div>
  );
}

/** A run or experiment that exists on the tracking server but is not Thomson.
 *
 *  Deliberately not an ErrorState: nothing failed. Someone followed a link to an
 *  ADEPT run, and saying so plainly beats "could not load this run", which reads
 *  as a bug in the browser. */
export function NotThomsonState({ message }: { message: string }) {
  return (
    <div className="state" role="status">
      <p className="state__title">Not a Thomson run</p>
      <p className="state__detail">{message}</p>
      <p className="state__detail">
        This is a browser for Thomson scattering analysis only. Runs from other projects on the
        same tracking server are not shown here — open it in the MLflow UI instead.
      </p>
    </div>
  );
}
