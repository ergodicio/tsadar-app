"""Write the browser API's OpenAPI schema to disk.

The schema is the contract between the two halves: the TypeScript client is
generated from it, and CI regenerates both to check they still agree. Keeping the
dump in a script rather than fetching from a running server means the check needs
no live MLflow and no port.

Usage::

    python scripts/dump_openapi.py [output_path]
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT / "frontend" / "src" / "api" / "openapi.json"

# npm runs this from frontend/, so the repo root is not on sys.path by default.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    # Imported here so a bad environment produces a clear traceback rather than
    # an import error at argument-parsing time.
    from tsadar_browser.app import create_app

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)

    schema = create_app().openapi()
    # sort_keys so the file is stable across runs and the CI diff is meaningful.
    output.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")

    print(f"wrote {output} ({len(schema.get('paths', {}))} paths)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
