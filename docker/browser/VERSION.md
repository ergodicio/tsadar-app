# Browser image versioning

The image is tagged `thomson-browser-v<VERSION>` from the repository-root
`VERSION` file, and the tag is **immutable**: pushing a tag that already exists in
ECR fails the workflow rather than overwriting it.

That is deliberate. continuum-infra's `continuum.yaml` pins image tags, and a
floating tag (`latest`, `latest-main`) means a pinned deployment silently changes
underneath itself — the failure mode its header comment documents. An immutable
tag makes "which image is deployed" answerable from the pin alone.

So: **bump `VERSION` in the same PR as the change you want deployed.** The
workflow will refuse to push otherwise, which is the intended nudge rather than an
obstacle.

The workflow's refusal is a check-then-push, so it cannot close the window between
the two by itself. It reliably catches the case it exists for — a forgotten
`VERSION` bump — but the actual enforcement wants `imageTagMutability: IMMUTABLE`
on the ECR repository, which is a registry-side change shared with the Streamlit
and runner images. See the deployment section of `docs/browser.md`.

A tag denoting one image also requires that a build of that `VERSION` is
reproducible, which is why `requirements-browser.txt` is pinned all the way
through its transitive dependencies.

After a push, hand the digest to infra (see ergodicio/continuum-infra#105 for the
pin-bump automation this slots into). The digest, not the tag, is what identifies
the image beyond doubt.
