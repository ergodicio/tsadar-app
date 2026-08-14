/**
 * Cursor-paginated run loading.
 *
 * MLflow paginates by opaque token, so this accumulates pages rather than
 * jumping to an offset: there is no way to ask the backend for "page 4".
 *
 * That decision makes pagination failures costlier than they look, which is why
 * two things here are deliberate:
 *
 * - A failed "Load more" is reported separately from a failed initial load.
 *   Because the cursor is intentionally not in the URL, there is nothing to
 *   resume from -- discarding accumulated pages means re-scrolling from page 1,
 *   so a transient 502 on page 5 must not take the table with it.
 * - A "Load more" response that arrives after the filters changed is dropped.
 *   Appending it would mix results from two different queries, and the stray row
 *   is real data, so it reads as a run that doesn't match your filter rather
 *   than as a bug.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api, type RunSummary } from "../api/client";
import type { RunFilters } from "../lib/urlState";

const PAGE_SIZE = 50;

export interface RunsState {
  runs: RunSummary[];
  loading: boolean;
  loadingMore: boolean;
  /** Failure of the initial load: the table has nothing to show. */
  error: string | null;
  /** Failure of a subsequent page: the table keeps what it already has. */
  loadMoreError: string | null;
  hasMore: boolean;
  loadMore: () => void;
  reload: () => void;
}

function messageFor(cause: unknown, fallback: string): string {
  return cause instanceof ApiError ? cause.message : fallback;
}

export function useRuns(filters: RunFilters): RunsState {
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadMoreError, setLoadMoreError] = useState<string | null>(null);
  const [nextToken, setNextToken] = useState<string | null>(null);
  const [reloadCount, setReloadCount] = useState(0);

  // Serialized so the effect re-runs on a filter *value* change rather than on
  // every render that happens to build a new object.
  const key = JSON.stringify(filters);

  const abortRef = useRef<AbortController | null>(null);
  // The query the currently displayed rows belong to. A page that resolves
  // against a stale key is discarded rather than appended.
  const keyRef = useRef(key);

  useEffect(() => {
    const controller = new AbortController();
    abortRef.current?.abort();
    abortRef.current = controller;
    keyRef.current = key;

    setLoading(true);
    setError(null);
    setLoadMoreError(null);

    api
      .runs({ ...filters, pageSize: PAGE_SIZE }, controller.signal)
      .then((page) => {
        setRuns(page.runs);
        setNextToken(page.next_page_token ?? null);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setRuns([]);
        setNextToken(null);
        setError(messageFor(cause, "Could not load runs."));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
    // filters is covered by `key`; reloadCount forces a manual retry.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reloadCount]);

  const loadMore = useCallback(() => {
    if (!nextToken || loadingMore) return;

    const requestKey = key;
    const signal = abortRef.current?.signal;

    setLoadingMore(true);
    setLoadMoreError(null);

    api
      .runs({ ...filters, pageSize: PAGE_SIZE, pageToken: nextToken }, signal)
      .then((page) => {
        // Both guards matter: the signal covers an in-flight abort, the key
        // covers a response that already resolved before the filters changed.
        if (signal?.aborted || keyRef.current !== requestKey) return;
        // Append rather than replace: the cursor only moves forward.
        setRuns((current) => [...current, ...page.runs]);
        setNextToken(page.next_page_token ?? null);
      })
      .catch((cause: unknown) => {
        if (signal?.aborted || keyRef.current !== requestKey) return;
        // Deliberately not `setError`: the accumulated pages stay on screen.
        setLoadMoreError(messageFor(cause, "Could not load more runs."));
      })
      .finally(() => {
        if (keyRef.current === requestKey) setLoadingMore(false);
      });
  }, [filters, key, nextToken, loadingMore]);

  const reload = useCallback(() => setReloadCount((count) => count + 1), []);

  return {
    runs,
    loading,
    loadingMore,
    error,
    loadMoreError,
    hasMore: nextToken !== null,
    loadMore,
    reload,
  };
}
