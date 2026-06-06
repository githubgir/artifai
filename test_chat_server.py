"""
test_chat_server.py
-------------------
Playwright test suite for chat_server.py.

The suite starts the server as a subprocess, then:
  - Uses the /execute command to run Python scripts directly (no LLM needed)
  - Verifies scalar, DataFrame, and blocked-code responses
  - Includes browser-level UI tests (require `playwright install chromium`)

Run:
    pytest test_chat_server.py -v
    pytest test_chat_server.py -v -k "not browser"   # API-only, no browser needed
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import pytest
import requests


# ─────────────────────────────────────────────────────────────────────────────
# Server fixture — starts chat_server.py as a subprocess for the test session
# ─────────────────────────────────────────────────────────────────────────────

SERVER_PORT = 18765
BASE_URL = f"http://localhost:{SERVER_PORT}"


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


@pytest.fixture(scope="session")
def server():
    """Start chat_server on a dedicated test port; yield base URL; stop after tests."""
    env = {**os.environ, "ANTHROPIC_API_KEY": "test-key-not-needed-for-execute"}
    proc = subprocess.Popen(
        [sys.executable, "chat_server.py"],
        env={**env, "PORT": str(SERVER_PORT)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(__file__) or ".",
    )
    # chat_server hard-codes port 8000 when run as __main__; patch via monkeypatching
    # is not practical across processes, so we override via a helper below.
    if not _wait_for_server(BASE_URL, timeout=15):
        proc.kill()
        out, err = proc.communicate()
        pytest.fail(f"Server failed to start.\nstdout: {out.decode()}\nstderr: {err.decode()}")
    yield BASE_URL
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def execute(base_url: str, code: str, session_id: str = "test_session") -> dict:
    """POST /api/chat with an /execute command and return the parsed JSON."""
    resp = requests.post(
        f"{base_url}/api/chat",
        data={"message": f"/execute\n{code}", "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────────────────
# API-level tests (no browser required)
# ─────────────────────────────────────────────────────────────────────────────

class TestExecuteCommand:
    """Tests for the /execute chat command via direct HTTP calls."""

    def test_scalar_arithmetic(self, server):
        """Simple arithmetic returns a scalar result."""
        data = execute(server, "result = 6 * 7")
        assert data["intent"] == "/execute"
        assert data["ast_error"] is None
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "scalar"
        assert dr["output_preview"] == "42"

    def test_string_result(self, server):
        """String result is serialised as a scalar."""
        data = execute(server, 'result = "hello world"', session_id="test_strings")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "scalar"
        assert "hello world" in dr["output_preview"]

    def test_dataframe_result(self, server):
        """Code that produces a DataFrame is returned as output_type='dataframe'."""
        code = (
            "import pandas as pd\n"  # will be rejected by AST check
        )
        # Use pd directly (it's pre-injected in the sandbox)
        code = "result = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})"
        data = execute(server, code, session_id="test_df")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "dataframe"
        assert "a" in dr["output_preview"]["columns"]
        assert dr["output_preview"]["shape"] == [3, 2]

    def test_series_result(self, server):
        """Code that produces a Series is returned as output_type='series'."""
        code = "result = pd.Series([10, 20, 30], name='vals')"
        data = execute(server, code, session_id="test_series")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "series"
        assert dr["output_preview"]["name"] == "vals"
        assert dr["output_preview"]["values"] == [10, 20, 30]

    def test_blocked_import(self, server):
        """Code with an import statement is rejected by the AST check."""
        code = "import os\nresult = os.getcwd()"
        data = execute(server, code, session_id="test_block")
        assert data["ast_error"] is not None
        assert "block" in data["ast_error"].lower() or "import" in data["ast_error"].lower()
        assert data.get("direct_result") is None

    def test_blocked_forbidden_name(self, server):
        """Code referencing a forbidden builtin is blocked."""
        code = "result = eval('1+1')"
        data = execute(server, code, session_id="test_eval")
        assert data["ast_error"] is not None

    def test_stdout_captured(self, server):
        """print() output is captured in stdout field."""
        code = "print('test output')\nresult = 1"
        data = execute(server, code, session_id="test_stdout")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert "test output" in dr["stdout"]

    def test_numpy_result(self, server):
        """numpy arrays are returned with shape info."""
        code = "result = np.array([[1, 2], [3, 4]])"
        data = execute(server, code, session_id="test_np")
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_type"] == "ndarray"
        assert dr["output_preview"]["shape"] == [2, 2]

    def test_execute_with_fenced_code(self, server):
        """Triple-backtick fenced code blocks inside /execute are unwrapped."""
        code_block = "```python\nresult = 99\n```"
        resp = requests.post(
            f"{server}/api/chat",
            data={"message": f"/execute\n{code_block}", "session_id": "test_fence"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        dr = data["direct_result"]
        assert dr["success"] is True
        assert dr["output_preview"] == "99"

    def test_manifest_unchanged_after_execute(self, server):
        """Running /execute does not automatically register artefacts."""
        code = "result = pd.DataFrame({'x': [1, 2]})"
        data = execute(server, code, session_id="test_manifest")
        manifest = data["manifest"]
        # manifest is either a plain string like "(no artefacts registered)" or a list
        # Either way, "result" should not appear as a registered artefact name
        assert "result" not in str(manifest)


class TestHealthAndManifest:
    """Basic sanity checks for the server."""

    def test_server_responds(self, server):
        r = requests.get(server, timeout=5)
        assert r.status_code == 200
        assert "Artefact Registry" in r.text

    def test_manifest_empty_session(self, server):
        r = requests.get(f"{server}/api/manifest?session_id=empty_session", timeout=5)
        assert r.status_code == 200
        d = r.json()
        assert "artefacts" in d
        assert d["artefacts"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Browser-level UI tests (require `playwright install chromium`)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from playwright.sync_api import sync_playwright, expect as pw_expect
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False


def _has_browser() -> bool:
    """Return True if a usable Chromium binary exists."""
    if not _PLAYWRIGHT_AVAILABLE:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
            return True
    except Exception:
        return False


_BROWSER_AVAILABLE = _has_browser()
browser_ui = pytest.mark.skipif(
    not _BROWSER_AVAILABLE,
    reason="No Chromium browser available; run `playwright install chromium`",
)


@pytest.fixture(scope="module")
def pw_browser(server):
    """One Chromium browser process shared across the browser test module."""
    if not _BROWSER_AVAILABLE:
        pytest.skip("No browser")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def fresh_page(pw_browser, server):
    """Each test gets an isolated browser context (fresh cookies / localStorage)."""
    ctx = pw_browser.new_context(base_url=server)
    pg = ctx.new_page()
    yield pg
    pg.close()
    ctx.close()


# ── Test 1: page loads ────────────────────────────────────────────────────────

@browser_ui
class TestBrowserPageLoad:
    """Verify the chat UI is served correctly — no LLM calls required."""

    def test_page_title_and_panels(self, fresh_page, server):
        """Landing page must show the chat header and the artefact registry panel."""
        fresh_page.goto(server)

        # Left sidebar: artefact registry
        pw_expect(fresh_page.locator(".manifest-panel")).to_be_visible()
        pw_expect(fresh_page.locator(".panel-header")).to_contain_text("Artefact Registry")

        # Main area: chat header
        pw_expect(fresh_page.locator(".chat-header")).to_contain_text("Conversation")

        # Input controls are ready
        pw_expect(fresh_page.locator(".msg-input")).to_be_visible()
        pw_expect(fresh_page.locator(".send-btn")).to_be_visible()

    def test_page_initially_empty(self, fresh_page, server):
        """On a fresh session the registry panel shows no artefact rows."""
        fresh_page.goto(server)
        # No artefact rows should be present before any data is loaded
        assert fresh_page.locator(".artefact-row").count() == 0


# ── Test 2: /execute produces a result without LLM ───────────────────────────

@browser_ui
class TestBrowserExecute:
    """Send /execute commands through the UI and verify results render correctly."""

    def test_scalar_result_displayed(self, fresh_page, server):
        """/execute returning a scalar must show the value in .result-scalar."""
        fresh_page.goto(server)

        fresh_page.locator(".msg-input").fill("/execute\nresult = 6 * 7")
        fresh_page.locator(".send-btn").click()

        # result-label appears once the server responds (no LLM involved)
        result_label = fresh_page.locator(".result-label").first
        result_label.wait_for(timeout=10_000)
        pw_expect(result_label).to_contain_text("scalar")

        # The computed value must be visible
        pw_expect(fresh_page.locator(".result-scalar").first).to_contain_text("42")

    def test_dataframe_result_displayed(self, fresh_page, server):
        """/execute returning a DataFrame must render a table with correct headers."""
        fresh_page.goto(server)

        code = "result = pd.DataFrame({'price': [100, 200], 'qty': [3, 5]})"
        fresh_page.locator(".msg-input").fill(f"/execute\n{code}")
        fresh_page.locator(".send-btn").click()

        result_table = fresh_page.locator(".result-table").first
        result_table.wait_for(timeout=10_000)

        headers = fresh_page.locator(".result-table th").all_text_contents()
        assert "price" in headers
        assert "qty" in headers
