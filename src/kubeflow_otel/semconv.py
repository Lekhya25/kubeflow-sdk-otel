"""Names of the spans, attributes and metrics this library emits.

Kept in one module so dashboards, alerts and tests share a single source of
truth. Metric names follow OpenTelemetry conventions (dot-separated, unit in
the ``unit`` field, durations in seconds). Prometheus renders them with
underscores and a unit suffix, e.g. ``kubeflow_client_operation_duration_seconds``.
"""

# ---------------------------------------------------------------- attributes
OPERATION = "kubeflow.operation"
BACKEND = "kubeflow.backend"
RUNTIME_NAME = "kubeflow.runtime.name"
TRAINER_TYPE = "kubeflow.trainer.type"
TRAINJOB_NAME = "kubeflow.trainjob.name"
TRAINJOB_STATUS = "kubeflow.trainjob.status"
TRAINJOB_PREVIOUS_STATUS = "kubeflow.trainjob.previous_status"
TRAINJOB_NUM_NODES = "kubeflow.trainjob.num_nodes"
WAIT_TARGET_STATUS = "kubeflow.wait.target_status"
WAIT_TIMEOUT = "kubeflow.wait.timeout"
WAIT_POLL_INTERVAL = "kubeflow.wait.polling_interval"
LOGS_STEP = "kubeflow.logs.step"
LOGS_FOLLOW = "kubeflow.logs.follow"
LOGS_LINES = "kubeflow.logs.lines"
RESULT_COUNT = "kubeflow.result.count"
ERROR_TYPE = "error.type"  # OpenTelemetry semantic convention

# -------------------------------------------------------------------- events
EVENT_STATUS_CHANGE = "kubeflow.trainjob.status_change"

# ------------------------------------------------------------------- metrics
METRIC_OPERATION_DURATION = "kubeflow.client.operation.duration"
METRIC_JOBS_SUBMITTED = "kubeflow.trainjob.submitted"
METRIC_JOBS_FINISHED = "kubeflow.trainjob.finished"
METRIC_JOB_DURATION = "kubeflow.trainjob.duration"
METRIC_JOB_STARTUP = "kubeflow.trainjob.startup.duration"

# Bucket boundaries in seconds. Client calls range from milliseconds (local
# lookups) to minutes (wait_for_job_status); jobs from seconds to hours.
OPERATION_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600,
)  # fmt: skip
JOB_DURATION_BUCKETS = (1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600, 7200, 21600, 86400)
JOB_STARTUP_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800)

# ------------------------------------------------------------ job statuses
# Mirrors kubeflow.trainer.constants (TRAINJOB_*); asserted equal in tests.
STATUS_RUNNING = "Running"
STATUS_COMPLETE = "Complete"
STATUS_FAILED = "Failed"
TERMINAL_STATUSES = frozenset({STATUS_COMPLETE, STATUS_FAILED})
