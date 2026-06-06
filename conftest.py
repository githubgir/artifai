"""conftest.py — project-wide pytest fixtures."""

import os
import subprocess
import sys
from pathlib import Path

# ── Chromium binary ───────────────────────────────────────────────────────────
_CHROMIUM_PATH = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
if os.path.exists(_CHROMIUM_PATH):
    os.environ.setdefault("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", _CHROMIUM_PATH)

# ── Local JS vendor libs (installed once, used to stub CDN in browser tests) ──
_VENDOR_DIR = Path(__file__).parent / ".js_vendor"

_CDN_MAP = {
    "https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.development.js":
        _VENDOR_DIR / "react.development.js",
    "https://cdnjs.cloudflare.com/ajax/libs/react-dom/18.2.0/umd/react-dom.development.js":
        _VENDOR_DIR / "react-dom.development.js",
    "https://cdnjs.cloudflare.com/ajax/libs/babel-standalone/7.23.2/babel.min.js":
        _VENDOR_DIR / "babel.min.js",
}


def _ensure_vendor() -> bool:
    """Install JS vendor libs via npm if missing. Returns True when available."""
    needed = [p for p in _CDN_MAP.values() if not p.exists()]
    if not needed:
        return True
    _VENDOR_DIR.mkdir(exist_ok=True)
    try:
        pkg_dir = _VENDOR_DIR / "_npm"
        subprocess.check_call(
            ["npm", "install", "--prefix", str(pkg_dir),
             "react@18.2.0", "react-dom@18.2.0", "@babel/standalone@7.23.2"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        nm = pkg_dir / "node_modules"
        (_VENDOR_DIR / "react.development.js").write_bytes(
            (nm / "react/umd/react.development.js").read_bytes()
        )
        (_VENDOR_DIR / "react-dom.development.js").write_bytes(
            (nm / "react-dom/umd/react-dom.development.js").read_bytes()
        )
        (_VENDOR_DIR / "babel.min.js").write_bytes(
            (nm / "@babel/standalone/babel.min.js").read_bytes()
        )
        return True
    except Exception:
        return False


_VENDOR_AVAILABLE = _ensure_vendor()


def setup_cdn_routes(page) -> None:
    """Intercept CDN script requests and serve local copies instead."""
    if not _VENDOR_AVAILABLE:
        return
    for url, local_path in _CDN_MAP.items():
        if local_path.exists():
            content = local_path.read_bytes()

            def make_handler(c):
                def handler(route):
                    route.fulfill(
                        status=200,
                        headers={"Content-Type": "application/javascript"},
                        body=c,
                    )
                return handler

            page.route(url, make_handler(content))
