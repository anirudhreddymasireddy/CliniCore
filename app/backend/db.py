"""Read-only Exasol access as MCP_READONLY: one shared connection, serialized by a lock.

pyexasol connections are not thread-safe and Exasol Personal allows only 20 parallel connections, so
the backend keeps exactly one session and reconnects once if it drops. The password comes from
CAREPRIORITY_DB_PASSWORD or, if unset, from the MCP sidecar config of the local deployment
(the same credential the MCP server uses). It is never logged or returned.
"""
import os
import re
import threading
from pathlib import Path

import pyexasol

DSN = os.environ.get("CAREPRIORITY_DB_DSN", "127.0.0.1:8563")
USER = "MCP_READONLY"
SIDECARS = Path.home() / ".exasol/personal/deployments/default/sidecars.yaml"

_lock = threading.Lock()
_conn = None


def _password():
    if os.environ.get("CAREPRIORITY_DB_PASSWORD"):
        return os.environ["CAREPRIORITY_DB_PASSWORD"]
    text = SIDECARS.read_text()
    user = re.search(r'- name: EXA_USER\s*\n\s*value: "([^"]+)"', text)
    pw = re.search(r'- name: EXA_PASSWORD\s*\n\s*value: "([^"]+)"', text)
    if not (user and pw and user.group(1) == USER):
        raise RuntimeError(f"No {USER} credential found; set CAREPRIORITY_DB_PASSWORD")
    return pw.group(1)


def _connect():
    return pyexasol.connect(dsn=DSN, user=USER, password=_password(), encryption=True,
                            websocket_sslopt={"cert_reqs": 0}, compression=True, fetch_dict=True)


def query(sql, params=None):
    """Run a read-only SELECT and return a list of dicts (keys upper-case)."""
    global _conn
    if not sql.lstrip().upper().startswith(("SELECT", "WITH")):
        raise ValueError("read-only backend: only SELECT / WITH queries are allowed")
    with _lock:
        for attempt in (1, 2):
            try:
                if _conn is None or _conn.is_closed:
                    _conn = _connect()
                return _conn.execute(sql, params).fetchall()
            except (pyexasol.ExaCommunicationError, pyexasol.ExaRuntimeError):
                _conn = None
                if attempt == 2:
                    raise


def current_user():
    return query("SELECT CURRENT_USER AS U FROM DUAL")[0]["U"]
