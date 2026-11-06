# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Perfetto trace processor wrapper."""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any

from profilelens.dtypes import QueryResult, TraceInfo

logger = logging.getLogger(__name__)

COMMON_TABLES = (
    "slice",
    "track",
    "thread",
    "process",
    "thread_track",
    "process_track",
    "counter",
    "counter_track",
    "sched_slice",
    "gpu_slice",
)

DEFAULT_LOAD_TIMEOUT = 300


class TraceProc:
    """Wrap Perfetto ``TraceProcessor`` for a trace file or a running server."""

    def __init__(self, trace_path: str | None = None, *, addr: str | None = None) -> None:
        if not trace_path and not addr:
            raise ValueError("Either trace_path or addr must be provided")
        if trace_path and addr:
            raise ValueError("Specify trace_path or addr, not both")

        self._addr: str | None = addr
        self._tmp_decompressed: str | None = None
        if trace_path:
            path = Path(trace_path)
            if not path.exists():
                raise FileNotFoundError(f"Trace file not found: {trace_path}")
            resolved = str(path.resolve())
            if resolved.endswith(".json.gz") or resolved.endswith(".trace.json.gz"):
                resolved = self._decompress_gz(resolved)
            self._trace_path: str | None = resolved
        else:
            self._trace_path = None
        self._tp: Any | None = None

    @staticmethod
    def _decompress_gz(gz_path: str) -> str:
        """Decompress a ``.json.gz`` trace to a temp file for Perfetto."""
        import gzip
        import tempfile

        stem = Path(gz_path).stem
        tmp = tempfile.NamedTemporaryFile(
            prefix="profilelens_", suffix=f"_{stem}", delete=False,
        )
        logger.info("Decompressing %s -> %s", gz_path, tmp.name)
        bytes_written = 0
        try:
            with gzip.open(gz_path, "rb") as fin:
                while True:
                    chunk = fin.read(64 * 1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    bytes_written += len(chunk)
        except (gzip.BadGzipFile, EOFError, OSError) as exc:
            logger.warning(
                "Gzip decompression incomplete (%s); recovered %.1f MB. "
                "Trace may be truncated.",
                exc, bytes_written / 1e6,
            )
        tmp.close()
        return tmp.name

    @property
    def trace_path(self) -> str:
        return self._trace_path or ""

    @property
    def server_addr(self) -> str | None:
        """HTTP address of the underlying trace_processor_shell, if loaded."""
        if self._addr:
            return self._addr
        if self._tp is None:
            return None
        try:
            conn = self._tp.http.conn
            return f"{conn.host}:{conn.port}"
        except AttributeError:
            return None

    @staticmethod
    def _bypass_proxy_for_localhost() -> dict[str, str | None]:
        """Temporarily allow localhost connections go direct."""
        saved: dict[str, str | None] = {}
        for key in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY",
                     "all_proxy", "ALL_PROXY"):
            saved[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
        no_proxy = os.environ.get("no_proxy", "")
        saved["no_proxy"] = no_proxy
        if "localhost" not in no_proxy and "127.0.0.1" not in no_proxy:
            parts = [p for p in no_proxy.split(",") if p]
            parts.extend(["localhost", "127.0.0.1", "::1"])
            os.environ["no_proxy"] = ",".join(parts)
        return saved

    @staticmethod
    def _restore_proxy_env(saved: dict[str, str | None]) -> None:
        """Restore after localhost connection is established."""
        for key, val in saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def _ensure_loaded(self) -> tuple[Any, str | None]:
        """Return the underlying TraceProcessor, loading on first call."""
        if self._tp is not None:
            return self._tp, None
        try:
            from perfetto.trace_processor import TraceProcessor as _TP
            from perfetto.trace_processor.api import TraceProcessorConfig

            if self._addr:
                self._tp = _TP(addr=self._addr)
                return self._tp, None

            bin_path = os.environ.get("TRACE_PROCESSOR_BIN")
            config = TraceProcessorConfig(
                bin_path=bin_path,
                load_timeout=DEFAULT_LOAD_TIMEOUT,
            )
            saved_proxy = self._bypass_proxy_for_localhost()
            try:
                self._tp = _TP(file_path=self._trace_path, config=config)
            finally:
                self._restore_proxy_env(saved_proxy)
            return self._tp, None
        except Exception as exc:
            msg = f"Failed to load trace: {exc}"
            logger.error(msg)
            return None, msg

    _QUERY_RETRIES = 5
    _QUERY_RETRY_BASE = 0.5

    def query(self, sql: str, label: str = "") -> tuple[QueryResult, str | None]:
        """Execute *sql* against the trace. Retries on transient connection errors."""
        tp, err = self._ensure_loaded()
        if err:
            return QueryResult(sql=sql, label=label, error=err), err

        last_exc: Exception | None = None
        retries = self._QUERY_RETRIES if self._addr else 1
        for attempt in range(retries):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = tp.query(sql)
                    df = result.as_pandas_dataframe()
                    rows = tuple(df.to_dict(orient="records"))
                cols = tuple(df.columns.tolist())
                return QueryResult(sql=sql, label=label, rows=rows, columns=cols), None
            except Exception as exc:
                last_exc = exc
                if attempt < retries - 1 and self._is_connection_error(exc):
                    delay = self._QUERY_RETRY_BASE * (2 ** attempt)
                    logger.debug("Query retry %d/%d in %.1fs: %s", attempt + 1, retries, delay, exc)
                    time.sleep(delay)
                    self._reconnect()
                    continue
                break

        msg = f"SQL error: {last_exc}"
        logger.warning(msg)
        return QueryResult(sql=sql, label=label, error=msg), msg

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in ("connection refused", "connection reset", "broken pipe", "errno 111"))

    def _reconnect(self) -> None:
        """Re-establish the HTTP connection to the shared server."""
        if self._tp is None or self._addr is None:
            return
        try:
            import http.client
            host, _, port = self._addr.partition(":")
            self._tp.http.conn = http.client.HTTPConnection(host, int(port))
        except Exception:
            pass

    def list_tables(self) -> tuple[list[str], str | None]:
        """Return table names present in the trace."""
        tp, err = self._ensure_loaded()
        if err:
            return [], err
        try:
            result = tp.query(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            df = result.as_pandas_dataframe()
            return df["name"].tolist(), None
        except Exception as exc:
            found: list[str] = []
            for t in COMMON_TABLES:
                try:
                    tp.query(f"SELECT 1 FROM {t} LIMIT 1")
                    found.append(t)
                except Exception:
                    continue
            if found:
                return found, None
            return [], f"Failed to list tables: {exc}"

    def get_schema(self, table: str) -> tuple[list[dict[str, str]], str | None]:
        """Return column info ``[{"name": ..., "type": ...}, ...]`` for *table*."""
        tp, err = self._ensure_loaded()
        if err:
            return [], err
        try:
            result = tp.query(f"PRAGMA table_info({table})")
            df = result.as_pandas_dataframe()
            cols = [
                {"name": row["name"], "type": row.get("type", "unknown")}
                for _, row in df.iterrows()
            ]
            return cols, None
        except Exception as exc:
            return [], f"Schema error for '{table}': {exc}"

    def trace_info(self) -> tuple[TraceInfo | None, str | None]:
        """Return high-level metadata about the loaded trace."""
        tables, err = self.list_tables()
        if err:
            return None, err
        row_count = 0
        tp, _ = self._ensure_loaded()
        if tp is not None:
            try:
                result = tp.query("SELECT COUNT(*) as cnt FROM slice")
                df = result.as_pandas_dataframe()
                row_count = int(df["cnt"].iloc[0])
            except Exception:
                pass
        return TraceInfo(
            path=self._trace_path,
            tables=tuple(tables),
            row_count=row_count,
        ), None
