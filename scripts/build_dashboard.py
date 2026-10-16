"""Generate the Grafana dashboard JSON (dashboards as code).

    python scripts/build_dashboard.py          # writes deploy/grafana/dashboards/kubeflow-sdk.json
    python scripts/build_dashboard.py --check  # fails if the committed JSON is out of date

Why generate instead of hand-editing JSON: every counter query below uses the
same "start-inclusive increase" expression, which is easy to get subtly wrong
when copy-pasted a dozen times.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "deploy/grafana/dashboards/kubeflow-sdk.json"
PROM = {"type": "prometheus", "uid": "prometheus"}
JAEGER = {"type": "jaeger", "uid": "jaeger"}


def increments(selector: str, window: str) -> str:
    """Counter increments inside ``window``, including a new series' first increment.

    ``increase()`` needs two samples, so for a counter whose series *starts*
    inside the window (every short-lived notebook or script) it misses the
    first increment entirely: a process that submits one job contributes 0.
    Each client process has its own series (unique ``instance``), so we take
    the value at the end of the window minus the value at its start, or minus
    0 when the series did not exist yet.
    """
    end = f"last_over_time({selector}[{window}])"
    start = f"last_over_time({selector}[{window}] offset {window})"  # offset binds to the selector
    return f"({end} - ({start} or {end} * 0))"


def p_quantile(q: float, bucket_selector: str, by: str, window: str) -> str:
    group = f"le, {by}" if by else "le"
    return f"histogram_quantile({q}, sum by ({group}) ({increments(bucket_selector, window)}))"


class Layout:
    """Places panels left-to-right on Grafana's 24-column grid."""

    def __init__(self) -> None:
        self.x = self.y = self.row_h = 0
        self.panels: list[dict] = []
        self._id = 0

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def row(self, title: str) -> None:
        self.newline()
        self.panels.append(
            {
                "type": "row",
                "id": self._next_id(),
                "title": title,
                "collapsed": False,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": self.y},
                "panels": [],
            }
        )
        self.y += 1

    def newline(self) -> None:
        self.y += self.row_h
        self.x = self.row_h = 0

    def add(self, panel: dict, w: int, h: int) -> None:
        if self.x + w > 24:
            self.newline()
        panel["id"] = self._next_id()
        panel["gridPos"] = {"h": h, "w": w, "x": self.x, "y": self.y}
        self.panels.append(panel)
        self.x += w
        self.row_h = max(self.row_h, h)


def target(
    expr: str, legend: str = "", ref: str = "A", instant: bool = False, exemplar: bool = False
) -> dict:
    return {
        "datasource": PROM,
        "refId": ref,
        "expr": expr,
        "legendFormat": legend or "__auto",
        "instant": instant,
        "range": not instant,
        "exemplar": exemplar,
    }


def stat(
    title: str,
    targets: list[dict],
    unit: str = "short",
    description: str = "",
    thresholds: list[tuple[float | None, str]] | None = None,
    decimals: int | None = None,
) -> dict:
    steps = [{"value": v, "color": c} for v, c in (thresholds or [(None, "blue")])]
    defaults: dict = {
        "unit": unit,
        "thresholds": {"mode": "absolute", "steps": steps},
        "color": {"mode": "thresholds"},
        "noValue": "0",
    }
    if decimals is not None:
        defaults["decimals"] = decimals
    return {
        "type": "stat",
        "title": title,
        "description": description,
        "datasource": PROM,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "none",
            "textMode": "auto",
            "justifyMode": "auto",
        },
    }


def timeseries(
    title: str,
    targets: list[dict],
    unit: str = "short",
    description: str = "",
    bars: bool = False,
    stack: bool = False,
) -> dict:
    custom: dict = {
        "drawStyle": "bars" if bars else "line",
        "lineWidth": 2,
        "fillOpacity": 60 if bars else 10,
        "showPoints": "never" if not bars else "auto",
        "spanNulls": True,
    }
    if stack:
        custom["stacking"] = {"mode": "normal", "group": "A"}
    return {
        "type": "timeseries",
        "title": title,
        "description": description,
        "datasource": PROM,
        "interval": "15s",
        "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "custom": custom, "color": {"mode": "palette-classic"}},
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
    }


def bargauge(title: str, targets: list[dict], unit: str = "short", description: str = "") -> dict:
    return {
        "type": "bargauge",
        "title": title,
        "description": description,
        "datasource": PROM,
        "targets": targets,
        "fieldConfig": {
            "defaults": {"unit": unit, "color": {"mode": "continuous-BlYlRd"}, "min": 0},
            "overrides": [],
        },
        "options": {
            "orientation": "horizontal",
            "displayMode": "gradient",
            "showUnfilled": True,
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
        },
    }


def build() -> dict:
    lay = Layout()
    op_count = "kubeflow_client_operation_duration_seconds_count"
    op_bucket = "kubeflow_client_operation_duration_seconds_bucket"
    submitted = "kubeflow_trainjob_submitted_total"
    finished = "kubeflow_trainjob_finished_total"
    completed = finished + '{kubeflow_trainjob_status="Complete"}'
    failed = finished + '{kubeflow_trainjob_status="Failed"}'
    op_errors = op_count + '{error_type!=""}'
    job = 'kubeflow_trainjob_name=~"$trainjob"'
    epoch_sum = "kubeflow_training_epoch_duration_seconds_sum{" + job + "}"
    epoch_count = "kubeflow_training_epoch_duration_seconds_count{" + job + "}"

    # ------------------------------------------------------------------ client
    lay.row("Client (kubeflow-sdk): what the SDK calls are doing")
    lay.add(
        stat(
            "Jobs submitted",
            [target(f"sum({increments(submitted, '$__range')})", instant=True)],
            description="TrainJobs submitted with TrainerClient.train() in the time range.",
        ),
        4,
        4,
    )
    lay.add(
        stat(
            "Jobs completed",
            [target(f"sum({increments(completed, '$__range')})", instant=True)],
            thresholds=[(None, "green")],
            description="Jobs observed reaching Complete (counted once per job).",
        ),
        4,
        4,
    )
    lay.add(
        stat(
            "Jobs failed",
            [target(f"sum({increments(failed, '$__range')})", instant=True)],
            thresholds=[(None, "green"), (1, "red")],
            description="Jobs observed reaching Failed.",
        ),
        4,
        4,
    )
    lay.add(
        stat(
            "Job success rate",
            [
                target(
                    f"sum({increments(completed, '$__range')})"
                    f" / sum({increments(finished, '$__range')})",
                    instant=True,
                )
            ],
            unit="percentunit",
            decimals=1,
            thresholds=[(None, "red"), (0.8, "orange"), (0.95, "green")],
        ),
        4,
        4,
    )
    lay.add(
        stat(
            "Client call error rate",
            [
                target(
                    f"sum({increments(op_errors, '$__range')})"
                    f" / sum({increments(op_count, '$__range')})",
                    instant=True,
                )
            ],
            unit="percentunit",
            decimals=1,
            thresholds=[(None, "green"), (0.01, "orange"), (0.05, "red")],
            description="Share of TrainerClient calls that raised.",
        ),
        4,
        4,
    )
    lay.add(
        stat(
            "Job duration p50 / p95",
            [
                target(
                    p_quantile(0.5, "kubeflow_trainjob_duration_seconds_bucket", "", "$__range"),
                    "p50",
                    "A",
                    True,
                ),
                target(
                    p_quantile(0.95, "kubeflow_trainjob_duration_seconds_bucket", "", "$__range"),
                    "p95",
                    "B",
                    True,
                ),
            ],
            unit="s",
            decimals=1,
            description="Creation to observed terminal status (as observed by the client).",
        ),
        4,
        4,
    )

    lay.add(
        timeseries(
            "Client call latency p95 by operation",
            [
                target(
                    p_quantile(0.95, op_bucket, "kubeflow_operation", "$__rate_interval"),
                    "{{kubeflow_operation}}",
                    exemplar=True,
                )
            ],
            unit="s",
            description="Dots are exemplars: click one to open that exact call's trace in Jaeger.",
        ),
        12,
        8,
    )
    lay.add(
        timeseries(
            "Client calls by operation",
            [
                target(
                    f"sum by (kubeflow_operation) ({increments(op_count, '$__interval')})",
                    "{{kubeflow_operation}}",
                )
            ],
            bars=True,
            stack=True,
        ),
        12,
        8,
    )
    lay.add(
        timeseries(
            "Jobs submitted and finished",
            [
                target(f"sum({increments(submitted, '$__interval')})", "submitted", "A"),
                target(
                    f"sum by (kubeflow_trainjob_status) ({increments(finished, '$__interval')})",
                    "{{kubeflow_trainjob_status}}",
                    "B",
                ),
            ],
            bars=True,
        ),
        12,
        8,
    )
    lay.add(
        timeseries(
            "Job startup p95 (created to running)",
            [
                target(
                    p_quantile(
                        0.95,
                        "kubeflow_trainjob_startup_duration_seconds_bucket",
                        "kubeflow_backend",
                        "$__rate_interval",
                    ),
                    "{{kubeflow_backend}}",
                )
            ],
            unit="s",
            description="Scheduling and queueing time before a job starts running.",
        ),
        12,
        8,
    )

    # ---------------------------------------------------------------- training
    lay.row("Training jobs (kubeflow-trainjob): what happens inside each job")
    lay.add(
        timeseries(
            "Training loss",
            [
                target(
                    f'kubeflow_training_metric{{kubeflow_training_metric_name="loss", {job}}}',
                    "{{kubeflow_trainjob_name}}",
                )
            ],
        ),
        12,
        9,
    )
    lay.add(
        timeseries(
            "Training accuracy",
            [
                target(
                    f'kubeflow_training_metric{{kubeflow_training_metric_name="accuracy", {job}}}',
                    "{{kubeflow_trainjob_name}}",
                )
            ],
            unit="percentunit",
        ),
        12,
        9,
    )
    lay.add(
        bargauge(
            "Current epoch",
            [
                target(
                    "max by (kubeflow_trainjob_name) "
                    f"(last_over_time(kubeflow_training_epoch{{{job}}}[$__range]))",
                    "{{kubeflow_trainjob_name}}",
                    instant=True,
                )
            ],
        ),
        12,
        8,
    )
    lay.add(
        bargauge(
            "Mean epoch time",
            [
                target(
                    f"sum by (kubeflow_trainjob_name) ({increments(epoch_sum, '$__range')})"
                    f" / sum by (kubeflow_trainjob_name) ({increments(epoch_count, '$__range')})",
                    "{{kubeflow_trainjob_name}}",
                    instant=True,
                )
            ],
            unit="s",
            description="Slow outliers here are stragglers worth a look in Jaeger.",
        ),
        12,
        8,
    )

    # ------------------------------------------------------------------ traces
    lay.row("Traces")
    lay.add(
        {
            "type": "table",
            "title": "Recent train() traces (click a trace ID to open it)",
            "datasource": JAEGER,
            "targets": [
                {
                    "datasource": JAEGER,
                    "refId": "A",
                    "queryType": "search",
                    "service": "kubeflow-sdk",
                    "operation": "kubeflow.trainer.train",
                    "limit": 20,
                }
            ],
            "fieldConfig": {"defaults": {}, "overrides": []},
            "options": {"showHeader": True},
        },
        24,
        9,
    )

    return {
        "uid": "kubeflow-sdk",
        "title": "Kubeflow SDK",
        "description": "Client calls, TrainJob lifecycle and in-job training metrics.",
        "tags": ["kubeflow", "opentelemetry"],
        "editable": True,
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-30m", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "graphTooltip": 1,
        "links": [
            {
                "title": "Jaeger",
                "type": "link",
                "url": "http://localhost:16686",
                "targetBlank": True,
            }
        ],
        "templating": {
            "list": [
                {
                    "name": "trainjob",
                    "label": "Training job",
                    "type": "query",
                    "datasource": PROM,
                    "query": {
                        "query": "label_values(kubeflow_training_metric, kubeflow_trainjob_name)",
                        "refId": "trainjob",
                    },
                    "definition": "label_values(kubeflow_training_metric, kubeflow_trainjob_name)",
                    "refresh": 2,
                    "includeAll": True,
                    "multi": True,
                    "allValue": ".*",
                    "current": {"selected": True, "text": ["All"], "value": ["$__all"]},
                    "sort": 1,
                }
            ]
        },
        "annotations": {"list": []},
        "panels": lay.panels,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed JSON is stale")
    args = parser.parse_args()
    rendered = json.dumps(build(), indent=2) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != rendered:
            print(f"{OUT} is out of date; run: python scripts/build_dashboard.py")
            return 1
        print("dashboard is up to date")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(rendered, encoding="utf-8", newline="\n")  # LF on every OS
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
