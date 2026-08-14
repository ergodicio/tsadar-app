"""Serving the built SPA alongside the API (issue #34).

The two cases that matter are the ones a plain static mount gets wrong: a
client-side route must return the app shell, and an unknown ``/api`` path must
stay a JSON 404.
"""

import pytest
from fastapi.testclient import TestClient

from tsadar_browser.app import create_app
from tsadar_browser.deps import get_gateway
from tsadar_browser.settings import Settings, get_settings
from tsadar_browser.spa import mount_spa

INDEX_HTML = '<!doctype html><html><body><div id="root"></div></body></html>'


@pytest.fixture
def static_dir(tmp_path):
    """A minimal stand-in for `npm run build` output."""
    dist = tmp_path / "static"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(INDEX_HTML)
    (dist / "assets" / "index-abc123.js").write_text("console.log('app')")
    (dist / "favicon.ico").write_bytes(b"\x00\x00\x01\x00")
    return dist


@pytest.fixture
def spa_client(gateway, static_dir):
    get_settings.cache_clear()
    app = create_app()
    mount_spa(app, static_dir)
    app.dependency_overrides[get_gateway] = lambda: gateway
    return TestClient(app)


class TestSpaServing:
    def test_root_serves_the_app_shell(self, spa_client):
        response = spa_client.get("/")
        assert response.status_code == 200
        assert '<div id="root">' in response.text

    def test_client_side_route_serves_the_shell(self, spa_client):
        """/runs/abc123 is resolved by the router, not by a file on disk.

        A plain static mount would 404 this, breaking every shared deep link.
        """
        response = spa_client.get("/runs/abc123")
        assert response.status_code == 200
        assert '<div id="root">' in response.text

    def test_nested_client_side_route_serves_the_shell(self, spa_client):
        assert '<div id="root">' in spa_client.get("/compare?runs=a,b").text

    def test_real_files_are_served_as_themselves(self, spa_client):
        response = spa_client.get("/favicon.ico")
        assert response.status_code == 200
        assert response.content.startswith(b"\x00\x00\x01\x00")

    def test_hashed_assets_are_served(self, spa_client):
        response = spa_client.get("/assets/index-abc123.js")
        assert response.status_code == 200
        assert "console.log" in response.text

    def test_index_is_not_cached(self, spa_client):
        """A cached shell would keep pointing at a previous image's assets."""
        assert "no-cache" in spa_client.get("/runs/abc").headers.get("cache-control", "")


class TestCaching:
    """The two halves of the bundle need opposite caching, and getting either
    wrong is invisible: the app still works, it is just slow or stale."""

    def test_hashed_assets_are_immutable(self, spa_client):
        # Safe only because the filename is content-hashed, which is what makes a
        # year-long cache correct rather than a stale-asset bug waiting to happen.
        cache_control = spa_client.get("/assets/index-abc123.js").headers["cache-control"]
        assert "immutable" in cache_control
        assert "max-age=31536000" in cache_control

    def test_the_shell_is_never_cached(self, spa_client):
        # The opposite of the assets: this is the one file whose URL does not
        # change between images, so caching it pins the browser to the old bundle.
        for path in ("/", "/runs/abc123", "/compare"):
            cache_control = spa_client.get(path).headers.get("cache-control", "")
            assert "no-store" in cache_control, f"{path} was cacheable: {cache_control!r}"

    def test_unhashed_files_are_not_marked_immutable(self, spa_client):
        # favicon.ico keeps its name across builds, so it must not be pinned for a
        # year the way a content-hashed asset can be.
        cache_control = spa_client.get("/favicon.ico").headers.get("cache-control", "")
        assert "immutable" not in cache_control


class TestApiIsNeverShadowed:
    def test_unknown_api_path_is_a_json_404_not_the_shell(self, spa_client):
        """Returning the shell here would make a typo'd endpoint a 200 of HTML,
        so the client would fail while parsing rather than on the status code."""
        response = spa_client.get("/api/nonexistent")
        assert response.status_code == 404
        assert '<div id="root">' not in response.text

    def test_bare_api_prefix_is_also_a_404(self, spa_client):
        assert spa_client.get("/api").status_code == 404

    def test_real_api_routes_still_work(self, spa_client):
        response = spa_client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_openapi_is_still_served(self, spa_client):
        assert spa_client.get("/api/openapi.json").status_code == 200

    def test_spa_route_is_absent_from_the_schema(self, spa_client):
        # It would otherwise appear as a documented endpoint and end up in the
        # generated client.
        paths = spa_client.get("/api/openapi.json").json()["paths"]
        assert not any("spa_path" in path for path in paths)


class TestTraversal:
    @pytest.mark.parametrize("path", ["../requirements-browser.txt", "assets/../../setup.py"])
    def test_escaping_the_static_dir_falls_back_to_the_shell(self, spa_client, path):
        # Not a file inside the bundle, so it is treated as a client-side route
        # rather than resolved outside it.
        response = spa_client.get(f"/{path}")
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            assert '<div id="root">' in response.text


class TestWithoutABundle:
    def test_api_only_when_there_is_no_bundle(self, gateway, tmp_path):
        """Development runs the SPA under Vite, so the API must work with no bundle."""
        get_settings.cache_clear()
        app = create_app()
        assert mount_spa(app, tmp_path / "absent") is False

        app.dependency_overrides[get_gateway] = lambda: gateway
        client = TestClient(app)
        assert client.get("/api/health").status_code == 200
        # No catch-all was registered, so a client-side route is a plain 404.
        assert client.get("/runs/abc").status_code == 404

    def test_settings_default_to_no_static_dir(self):
        get_settings.cache_clear()
        assert Settings().static_dir is None
