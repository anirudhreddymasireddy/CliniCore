#!/usr/bin/env bash
# Replay mode on live data: backend reads Exasol read-only as MCP_READONLY.
exec "$(cd "$(dirname "$0")" && pwd)/_start.sh" live "#/replay"
