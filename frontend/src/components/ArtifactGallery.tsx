/**
 * The PNG/CSV listing.
 *
 * Always shown, not only as a fallback: even a fully interactive run has images
 * the slicing API cannot reproduce (the distribution-function contours, the
 * error histogram, and the lineout plots that *do* include IRF and noise
 * components). For angular and pre-contract runs it is the whole view, which is
 * why the page must never be blank.
 */

import { artifactUrl, type ArtifactEntry } from "../api/client";

const IMAGE_PATTERN = /\.(png|jpe?g|svg|gif)$/i;

interface ArtifactGalleryProps {
  runId: string;
  artifacts: ArtifactEntry[];
  /** Shown above the gallery when it is standing in for the interactive views. */
  fallbackMessage?: string | null;
}

export function ArtifactGallery({ runId, artifacts, fallbackMessage }: ArtifactGalleryProps) {
  const files = artifacts.filter((entry) => !entry.is_dir);
  const images = files.filter((entry) => IMAGE_PATTERN.test(entry.path));
  const others = files.filter((entry) => !IMAGE_PATTERN.test(entry.path));

  return (
    <section className="panel" aria-labelledby="gallery-heading">
      <header className="panel__header">
        <h2 id="gallery-heading">Plots and files</h2>
        <span className="panel__meta">
          {images.length} image{images.length === 1 ? "" : "s"}, {others.length} other
        </span>
      </header>

      {fallbackMessage && (
        <p className="panel__status panel__status--notice" role="status">
          {fallbackMessage}
        </p>
      )}

      {files.length === 0 && <p className="panel__status">This run logged no artifacts.</p>}

      {images.length > 0 && (
        <ul className="gallery">
          {images.map((entry) => (
            <li key={entry.path} className="gallery__item">
              <a href={artifactUrl(runId, entry.path)} target="_blank" rel="noreferrer">
                <img src={artifactUrl(runId, entry.path)} alt={entry.path} loading="lazy" />
                <span className="gallery__caption">{entry.path}</span>
              </a>
            </li>
          ))}
        </ul>
      )}

      {others.length > 0 && (
        <ul className="filelist">
          {others.map((entry) => (
            <li key={entry.path}>
              <a href={artifactUrl(runId, entry.path)} target="_blank" rel="noreferrer">
                {entry.path}
              </a>
              {entry.size !== null && entry.size !== undefined && (
                <span className="filelist__size">{Math.max(1, Math.round(entry.size / 1024))} KB</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
