# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Result formatting for trace query output."""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.table import Table

from profilelens.dtypes import QueryResult


def _number_fmt(v: Any) -> str:
    """Formatting for numeric values."""
    if isinstance(v, float):
        return f"{v:,.4f}"
    return str(v)


def print_rich(results: list[QueryResult], console: Console | None = None) -> None:
    """Render results as tables to console."""
    con = console or Console()
    for qr in results:
        if qr.error:
            con.print(f"[bold red]Error[/] ({qr.label}): {qr.error}")
            continue

        table = Table(title=qr.label, show_lines=False, pad_edge=True)
        for col in qr.columns:
            table.add_column(col, justify="right" if col != "name" else "left")

        for row in qr.rows:
            table.add_row(*[_number_fmt(row.get(c, "")) for c in qr.columns])

        con.print(table)
        con.print()


def format_json(results: list[QueryResult]) -> str:
    """Return JSON representation of results."""
    payload = []
    for qr in results:
        entry: dict[str, Any] = {"label": qr.label, "sql": qr.sql}
        if qr.error:
            entry["error"] = qr.error
        else:
            entry["columns"] = list(qr.columns)
            entry["rows"] = [dict(r) for r in qr.rows]
            entry["count"] = len(qr.rows)
        payload.append(entry)
    return json.dumps(payload, indent=2, default=str)


def format_markdown(results: list[QueryResult]) -> str:
    """Return GitHub-like markdown tables."""
    parts: list[str] = []
    for qr in results:
        parts.append(f"### {qr.label}\n")
        if qr.error:
            parts.append(f"> **Error:** {qr.error}\n")
            continue
        if not qr.rows:
            parts.append("_No results._\n")
            continue
        parts.append("| " + " | ".join(qr.columns) + " |")
        parts.append("| " + " | ".join("---" for _ in qr.columns) + " |")
        for row in qr.rows:
            cells = [_number_fmt(row.get(c, "")) for c in qr.columns]
            parts.append("| " + " | ".join(cells) + " |")
        parts.append("")
    return "\n".join(parts)
