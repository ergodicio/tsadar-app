/**
 * The page a physicist lives on.
 *
 * Layout is chosen from the capability probe (`GET /api/runs/{id}/datasets`)
 * rather than guessed: a run that cannot be served interactively -- angular, or
 * predating the artifact contract -- gets the gallery with an honest explanation
 * of *why*, never a blank panel or an error that reads like a bug.
 *
 * The selected lineout lives in the URL alongside the spectrum, so a link can
 * point at one specific lineout of one specific run.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { ApiError, api, type DatasetAvailability, type RunDetail as RunDetailModel } from "../api/client";
import { ArtifactGallery } from "../components/ArtifactGallery";
import { ConfigPanel } from "../components/ConfigPanel";
import { LineoutPanel } from "../components/LineoutPanel";
import { LossPanel } from "../components/LossPanel";
import { ProfilesPanel } from "../components/ProfilesPanel";
import { RunHeader } from "../components/RunHeader";
import { SpectrogramPanel } from "../components/SpectrogramPanel";
import { ErrorState, LoadingState, NotThomsonState } from "../components/StateViews";

export function RunDetail() {
  const { runId = "" } = useParams();
  const [search, setSearch] = useSearchParams();

  const [run, setRun] = useState<RunDetailModel | null>(null);
  const [availability, setAvailability] = useState<DatasetAvailability | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [errorReason, setErrorReason] = useState<string | null>(null);
  const [reloadCount, setReloadCount] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    setErrorReason(null);
    setRun(null);
    setAvailability(null);

    // Both in parallel: the probe never errors for a legitimate run, so a
    // failure here means the run itself could not be read.
    Promise.all([api.run(runId, controller.signal), api.datasets(runId, controller.signal)])
      .then(([detail, probe]) => {
        setRun(detail);
        setAvailability(probe);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof ApiError ? cause.message : "Could not load this run.");
        setErrorReason(cause instanceof ApiError ? (cause.reason ?? null) : null);
      });

    return () => controller.abort();
  }, [runId, reloadCount]);

  const spectra = availability?.spectra ?? [];
  const which = search.get("which") ?? spectra[0]?.which ?? "ele";
  const lineoutIndex = Number(search.get("lineout") ?? "0") || 0;

  const spectrum = useMemo(
    () => spectra.find((candidate) => candidate.which === which) ?? spectra[0],
    [spectra, which],
  );

  const setParam = useCallback(
    (key: string, value: string) => {
      const next = new URLSearchParams(search);
      next.set(key, value);
      setSearch(next, { replace: true });
    },
    [search, setSearch],
  );

  const onWhichChange = useCallback(
    (nextWhich: string) => {
      // Lineout indices are per-spectrum, and ele/ion can have different counts,
      // so switching spectrum resets to the first lineout rather than carrying
      // over an index that may not exist.
      const next = new URLSearchParams(search);
      next.set("which", nextWhich);
      next.set("lineout", "0");
      setSearch(next, { replace: true });
    },
    [search, setSearch],
  );

  const onLineoutChange = useCallback(
    (index: number) => setParam("lineout", String(index)),
    [setParam],
  );

  // A non-Thomson run is not a failure, so it must not offer "try again": the
  // answer will not change on a retry.
  if (error && errorReason === "not_thomson") return <NotThomsonState message={error} />;
  if (error) {
    return (
      <ErrorState message={error} onRetry={() => setReloadCount((count) => count + 1)} />
    );
  }
  if (!run || !availability) return <LoadingState />;

  const interactive = availability.supported && spectrum !== undefined;

  return (
    <article className="detail">
      <RunHeader run={run} />

      {interactive ? (
        <>
          <SpectrogramPanel
            runId={runId}
            spectra={spectra}
            which={spectrum.which}
            onWhichChange={onWhichChange}
            lineoutIndex={lineoutIndex}
            onLineoutChange={onLineoutChange}
          />
          <LineoutPanel
            runId={runId}
            spectrum={spectrum}
            which={spectrum.which}
            index={lineoutIndex}
            onIndexChange={onLineoutChange}
          />
          {availability.profiles_available && (
            <ProfilesPanel
              runId={runId}
              lineoutIndex={lineoutIndex}
              onLineoutChange={onLineoutChange}
            />
          )}
        </>
      ) : (
        // No interactive views. The message comes from the backend's reason code,
        // so "angular run, out of scope" never reads as "no data found".
        <section className="panel">
          <h2>Interactive views unavailable</h2>
          <p className="panel__status panel__status--notice" role="status">
            {availability.message ?? "This run has no readable datasets."}
          </p>
        </section>
      )}

      <LossPanel runId={runId} run={run} />
      <ConfigPanel runId={runId} run={run} />
      <ArtifactGallery
        runId={runId}
        artifacts={run.artifacts}
        fallbackMessage={
          interactive
            ? null
            : "Showing the plots this run logged instead of the interactive panels."
        }
      />
    </article>
  );
}
