/**
 * Load several runs for comparison.
 *
 * Each run is fetched independently and a failure is recorded against that run
 * rather than failing the page: comparing four runs where one is unreadable should
 * still compare the other three, and say what happened to the fourth.
 */

import { useCallback, useEffect, useState } from "react";

import { ApiError, api, type Profiles } from "../api/client";
import { exclusionReason, type ComparisonRun } from "../lib/compare";

export interface ComparisonState {
  runs: ComparisonRun[];
  /** Runs that could not be loaded at all, as run id → message. */
  failures: Array<{ runId: string; message: string }>;
  loading: boolean;
  reload: () => void;
}

async function loadRun(runId: string, signal: AbortSignal): Promise<ComparisonRun> {
  const [detail, availability] = await Promise.all([
    api.run(runId, signal),
    api.datasets(runId, signal),
  ]);

  // Profiles are the overlay's substance, but their absence is not fatal -- the
  // run still contributes to the summary table and the config diff.
  let profiles: Profiles | null = null;
  let profilesError: string | null = null;
  if (availability.supported && availability.profiles_available) {
    profiles = await api.profiles(runId, signal).catch((cause: unknown) => {
      // An abort is not "this run has no profiles" -- it means the comparison
      // moved on while the request was in flight. Swallowing it would resolve
      // this run with a wrong exclusion reason, and the caller discards aborted
      // results anyway, so it is rethrown rather than recorded.
      if (signal.aborted) throw cause;
      profilesError = cause instanceof ApiError ? cause.message : "the request failed";
      return null;
    });
  }

  return {
    runId,
    detail,
    availability,
    profiles,
    excluded: exclusionReason(availability, profiles, profilesError),
  };
}

export function useComparison(runIds: string[]): ComparisonState {
  const [runs, setRuns] = useState<ComparisonRun[]>([]);
  const [failures, setFailures] = useState<Array<{ runId: string; message: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [reloadCount, setReloadCount] = useState(0);

  // Stable across renders so a consumer can pass it to a memoized child (the
  // retry button's `onRetry`) without defeating the memo.
  const reload = useCallback(() => setReloadCount((count) => count + 1), []);

  const key = runIds.join(",");

  useEffect(() => {
    if (runIds.length === 0) {
      setRuns([]);
      setFailures([]);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    setLoading(true);

    Promise.all(
      runIds.map((runId) =>
        loadRun(runId, controller.signal).then(
          (run) => ({ ok: true as const, run }),
          (cause: unknown) => ({
            ok: false as const,
            runId,
            message: cause instanceof ApiError ? cause.message : "Could not load this run.",
          }),
        ),
      ),
    )
      .then((results) => {
        if (controller.signal.aborted) return;
        setRuns(results.filter((result) => result.ok).map((result) => result.run));
        setFailures(
          results
            .filter((result) => !result.ok)
            .map((result) => ({ runId: result.runId, message: result.message })),
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
    // runIds is covered by `key`.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reloadCount]);

  return { runs, failures, loading, reload };
}
