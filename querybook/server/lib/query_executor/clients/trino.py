import logging
import time
from typing import Any, Dict, List, Optional

import trino
import trino.client
from trino.exceptions import TrinoUserError

from lib.query_executor.base_client import ClientBaseClass, CursorBaseClass
from lib.query_executor.clients.utils.presto_cursor import PrestoCursorMixin
from lib.query_executor.connection_string.trino import get_trino_connection_conf

LOG = logging.getLogger(__name__)

# Trino query states that indicate computation is done and the server
# is only streaming result rows back to the client.
_TRINO_FINISHING_STATE = "FINISHING"


class TrinoCursor(PrestoCursorMixin[trino.dbapi.Cursor, List[Any]], CursorBaseClass):
    """Non-blocking Trino cursor that supports incremental polling.

    The default ``trino.dbapi.Cursor.execute()`` blocks until at least
    one result row is available, which prevents the Querybook executor
    poll loop from emitting real-time progress updates (tracking URL,
    data scanned, percent complete) to the frontend.

    This cursor overrides ``run()`` to only send the initial SQL POST
    and return immediately.  The actual result fetching is driven by
    ``poll()``, which is called repeatedly by the executor loop with
    configurable sleep intervals in between.
    """

    def __init__(self, cursor: trino.dbapi.Cursor) -> None:
        self._cursor = cursor
        self._request = cursor._request
        self.rows: List[List[Any]] = []
        self._init_query_state_vars()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _init_query_state_vars(self) -> None:
        self.rows = []
        self._tracking_url = None
        self._percent_complete = 0
        self._physical_input_bytes = 0
        self._query_id = None

    @property
    def _log_prefix(self) -> str:
        return f"[Trino query_id={self._query_id}]"

    # ------------------------------------------------------------------
    # Query execution (non-blocking)
    # ------------------------------------------------------------------

    def run(self, query: str):
        """Submit the query to Trino without blocking for results.

        Replicates only the non-blocking portion of
        ``TrinoQuery.execute()`` — sends the SQL via POST, records the
        query-id and initial state, and returns immediately so that
        ``poll()`` can drive the rest one page at a time.
        """
        self._init_query_state_vars()

        request = self._cursor._request
        trino_query = trino.client.TrinoQuery(
            request,
            query=query,
            legacy_primitive_types=self._cursor._legacy_primitive_types,
        )
        self._cursor._query = trino_query

        response = request.post(query)
        status = request.process(response)

        trino_query._info_uri = status.info_uri
        trino_query._query_id = status.id
        trino_query._stats.update({"queryId": trino_query.query_id})
        trino_query._update_state(status)
        trino_query._warnings = getattr(status, "warnings", [])

        if status.next_uri is None:
            trino_query._finished = True

        trino_query._result = trino.client.TrinoResult(trino_query, [])
        self._cursor._iterator = iter([])

        self._query_id = trino_query.query_id
        LOG.info(
            "%s Query submitted, state=%s, finished=%s",
            self._log_prefix,
            self._cursor.stats.get("state"),
            trino_query.finished,
        )

    # ------------------------------------------------------------------
    # Incremental polling
    # ------------------------------------------------------------------

    def poll(self) -> bool:
        """Fetch one page of results and update query statistics.

        Returns ``True`` when the query is fully complete (all result
        pages consumed).

        When Trino signals that query computation is done (state is
        ``FINISHING`` or ``FINISHED``), all remaining result pages are
        drained in a tight loop to avoid unnecessary sleep delays.
        """
        try:
            query = self._cursor._query
            state = self._cursor.stats.get("state", "UNKNOWN")

            if not query.finished:
                LOG.info(
                    "%s Polling: state=%s, rows_so_far=%d, physicalInputBytes=%d",
                    self._log_prefix,
                    state,
                    len(self.rows),
                    self._cursor.stats.get("physicalInputBytes", 0),
                )

                # Fetch a single page from the Trino REST API.
                rows = query.fetch()
                self.rows.extend(rows)

                # If computation is done, drain all remaining pages at once.
                if self._cursor.stats.get("state") == _TRINO_FINISHING_STATE:
                    self._drain_remaining_results(query)

            self._cursor._iterator = iter(self.rows)
            self._update_stats(self._cursor.stats)

            completed = query.finished
            if completed:
                LOG.info(
                    "%s Completed: total_rows=%d, physicalInputBytes=%d",
                    self._log_prefix,
                    len(self.rows),
                    self._physical_input_bytes,
                )

        except TrinoUserError as e:
            LOG.error("%s TrinoUserError: %s", self._log_prefix, str(e))
            poll_result = {"queryId": e.query_id}
            self._update_tracking_url(poll_result)
            raise

        return completed

    def _drain_remaining_results(self, query) -> None:
        """Fetch all remaining result pages in a tight loop."""
        LOG.info(
            "%s Query execution finished (state=%s), draining remaining results...",
            self._log_prefix,
            self._cursor.stats.get("state"),
        )
        drain_start = time.time()
        page_count = 0
        while not query.finished:
            rows = query.fetch()
            self.rows.extend(rows)
            page_count += 1
        LOG.info(
            "%s Drained %d pages in %.2fs, total_rows=%d",
            self._log_prefix,
            page_count,
            time.time() - drain_start,
            len(self.rows),
        )

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def _update_stats(self, poll_result: Dict[str, Any]) -> None:
        """Update all tracked statistics from the latest poll result."""
        if not poll_result:
            return
        self._update_tracking_url(poll_result)
        self._update_percent_complete(poll_result)
        self._update_physical_input_bytes(poll_result)

    def _update_tracking_url(self, poll_result: Dict[str, Any]) -> None:
        if self._tracking_url is None:
            query_id = poll_result["queryId"]
            self._tracking_url = (
                f"{self._request._http_scheme}://"
                f"{self._request._host}:{self._request._port}"
                f"/ui/plan.html?{query_id}"
            )

    def _update_percent_complete(self, poll_result: Dict[str, Any]) -> None:
        self._percent_complete = poll_result.get("progressPercentage", 0)

    def _update_physical_input_bytes(self, poll_result: Dict[str, Any]) -> None:
        self._physical_input_bytes = poll_result.get("physicalInputBytes", 0)

    @property
    def physical_input_bytes(self) -> int:
        return self._physical_input_bytes


class TrinoClient(ClientBaseClass):
    def __init__(
        self,
        connection_string: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        proxy_user: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        trino_conf = get_trino_connection_conf(connection_string)

        host = trino_conf.host
        port = 8080 if not trino_conf.port else trino_conf.port

        auth = trino.constants.DEFAULT_AUTH
        if username is not None and password is not None:
            auth = trino.auth.BasicAuthentication(username, password)

        connection = trino.dbapi.connect(
            host=host,
            port=port,
            catalog=trino_conf.catalog,
            schema=trino_conf.schema,
            auth=auth,
            user=proxy_user if proxy_user else username,
            http_scheme=trino_conf.protocol,
        )
        self._connection = connection
        super(TrinoClient, self).__init__()

    def cursor(self) -> TrinoCursor:
        return TrinoCursor(cursor=self._connection.cursor())
