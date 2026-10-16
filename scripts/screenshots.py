"""Capture the README screenshots from a running stack (after the demo has run).

pip install playwright && playwright install chromium
python scripts/screenshots.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs/images"
GRAFANA = "http://localhost:3000"
JAEGER = "http://localhost:16686"


def latest_trace_id(operation: str) -> str:
    query = urllib.parse.urlencode(
        {"service": "kubeflow-sdk", "operation": operation, "limit": 1, "lookback": "2h"}
    )
    with urllib.request.urlopen(f"{JAEGER}/api/traces?{query}", timeout=10) as resp:
        traces = json.load(resp)["data"]
    if not traces:
        sys.exit(f"no '{operation}' trace found; run the demo first")
    return traces[0]["traceID"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trace_id = latest_trace_id("kubeflow.demo.sweep")
    with sync_playwright() as p:
        browser = p.chromium.launch()

        page = browser.new_page(viewport={"width": 1600, "height": 1300}, device_scale_factor=1)
        # usage: screenshots.py [WINDOW]  e.g. 45m (relative to now, should cover the demo)
        #        screenshots.py FROM TO    Grafana time values, e.g. epoch milliseconds
        args = sys.argv[1:]
        start, end = (
            (args[0], args[1]) if len(args) == 2 else (f"now-{(args or ['45m'])[0]}", "now")
        )
        page.goto(f"{GRAFANA}/d/kubeflow-sdk/kubeflow-sdk?orgId=1&from={start}&to={end}&kiosk")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(6000)  # let every panel finish its queries and render
        page.screenshot(path=OUT / "grafana-dashboard.png", full_page=True)

        page = browser.new_page(viewport={"width": 1600, "height": 1100}, device_scale_factor=1)
        page.goto(f"{JAEGER}/trace/{trace_id}")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(3000)
        # fold the epoch level: each job becomes one row with its epochs drawn as ticks
        # header icons are: expand +1, collapse +1, expand all, collapse all
        buttons = page.locator(".TimelineCollapser > *")
        if buttons.count() >= 2:
            buttons.nth(1).click()
            page.mouse.move(800, 1090)  # move off the button so its tooltip hides
            page.wait_for_timeout(1000)
        page.screenshot(path=OUT / "jaeger-sweep-trace.png", full_page=True)

        browser.close()
    print(f"saved screenshots to {OUT} (trace {trace_id})")


if __name__ == "__main__":
    main()
