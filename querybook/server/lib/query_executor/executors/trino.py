import time

from trino.exceptions import Error, TrinoQueryError

from const.query_execution import QueryExecutionErrorType
from lib.query_executor.base_executor import QueryExecutorBaseClass
from lib.query_executor.clients.trino import TrinoClient
from lib.query_executor.executor_template.templates import trino_executor_template
from lib.query_executor.utils import get_parsed_syntax_error

# -- Polling intervals (seconds) ------------------------------------------
_POLL_INTERVAL_SHORT = 2  # used for the first 30 seconds
_POLL_INTERVAL_DEFAULT = 5  # used between 30 and 60 seconds
_POLL_INTERVAL_LONG = 15  # used after 60 seconds


def _format_bytes(num_bytes: int) -> str:
    """Format a byte count into a human-readable string (e.g. ``1.50 GB``)."""
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} EB"


def _get_trino_error_dict(e):
    """Extract the error dict from a Trino exception, if present."""
    if hasattr(e, "args") and e.args[0] is not None:
        error_arg = e.args[0]
        if isinstance(error_arg, dict):
            return error_arg
    return None


class TrinoQueryExecutor(QueryExecutorBaseClass):
    @classmethod
    def _get_client(cls, client_setting):
        return TrinoClient(**client_setting)

    @classmethod
    def EXECUTOR_NAME(cls):
        return "trino"

    @classmethod
    def EXECUTOR_LANGUAGE(cls):
        return "trino"

    @classmethod
    def EXECUTOR_TEMPLATE(cls):
        return trino_executor_template

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _parse_exception(self, e):
        error_type = QueryExecutionErrorType.INTERNAL.value
        error_str = str(e)
        error_extracted = None

        if isinstance(e, TrinoQueryError):
            try:
                line_number, column_number = e.error_location
                return get_parsed_syntax_error(
                    e.message,
                    line_number - 1,
                    column_number - 1,
                )
            except Exception:
                return (
                    QueryExecutionErrorType.ENGINE.value,
                    e.message,
                    error_extracted,
                )

        if isinstance(e, Error):
            error_type = QueryExecutionErrorType.ENGINE.value
            try:
                error_dict = _get_trino_error_dict(e)
                if error_dict:
                    error_extracted = error_dict.get("message", None)
            except Exception:
                pass

        return error_type, error_str, error_extracted

    # ------------------------------------------------------------------
    # Meta info (tracking URL + data scanned)
    # ------------------------------------------------------------------

    @property
    def meta_info(self):
        """Build the statement meta string shown in the UI.

        Includes the Trino tracking URL and the amount of physical
        data scanned, formatted as a human-readable byte string.
        """
        parts = []
        if self._cursor.tracking_url:
            parts.append(f"Tracking Url: {self._cursor.tracking_url}")
        physical_input_bytes = self._cursor.physical_input_bytes
        if physical_input_bytes:
            parts.append(f"Data Scanned: {_format_bytes(physical_input_bytes)}")
        return "\n".join(parts) + "\n" if parts else ""

    # ------------------------------------------------------------------
    # Poll interval
    # ------------------------------------------------------------------

    def sleep(self):
        """Pause between poll cycles.

        Uses a shorter interval early on for responsiveness and longer
        intervals afterwards to reduce load on long-running queries.
        """
        elapsed = time.time() - self._start_time
        if elapsed < 30:
            interval = _POLL_INTERVAL_SHORT
        elif elapsed < 60:
            interval = _POLL_INTERVAL_DEFAULT
        else:
            interval = _POLL_INTERVAL_LONG
        time.sleep(interval)
