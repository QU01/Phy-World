"""F9: servidor MCP de World4 por stdio (JSON-RPC 2.0, un mensaje por línea), sin dependencias.

Métodos: ``initialize``, ``notifications/initialized``, ``ping``, ``tools/list`` y ``tools/call``. Las
herramientas son las de ``tools.py``. stdout es SÓLO del protocolo: todo lo que el modelo imprima se desvía
a stderr. Un error de una herramienta (dominio, argumentos) vuelve como resultado con ``isError``; un
método desconocido, como error JSON-RPC.

    python world4_cli.py --mcp           (o: world4 --mcp, o: python -m world4.mcp.server)
"""
from __future__ import annotations

import contextlib
import json
import sys
import traceback

PROTOCOL = "2025-06-18"


def _result(rid, result):
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _error(rid, code, message):
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    """Una petición -> una respuesta (None para notificaciones)."""
    from world4 import MODEL_REVISION, __version__
    from world4.mcp import tools
    rid = msg.get("id")
    method = msg.get("method")
    if not isinstance(method, str):
        return _error(rid, -32600, "invalid request")
    if method.startswith("notifications/"):
        return None
    if method == "initialize":
        asked = (msg.get("params") or {}).get("protocolVersion")
        return _result(rid, {"protocolVersion": asked or PROTOCOL, "capabilities": {"tools": {"listChanged": False}},
                             "serverInfo": {"name": "world4", "version": f"{__version__}+rev{MODEL_REVISION}"},
                             "instructions": "Phy-World (World4). Every number has a unit and a band; results "
                                             "beyond the measured horizon carry exploratory=true and are behaviour "
                                             "modes, not forecasts."})
    if method == "ping":
        return _result(rid, {})
    if method == "tools/list":
        return _result(rid, {"tools": [{"name": n, "description": d, "inputSchema": s}
                                       for n, (_, d, s) in tools.TOOLS.items()]})
    if method == "tools/call":
        p = msg.get("params") or {}
        name = p.get("name")
        if name not in tools.TOOLS:
            return _error(rid, -32602, f"unknown tool {name!r}")
        try:
            with contextlib.redirect_stdout(sys.stderr):
                out = tools.call(name, p.get("arguments") or {})
            text = json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            return _result(rid, {"content": [{"type": "text", "text": text}], "structuredContent": json.loads(text),
                                 "isError": False})
        except Exception as e:                       # noqa: BLE001  (vuelve como resultado de error)
            traceback.print_exc(file=sys.stderr)
            return _result(rid, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True})
    return _error(rid, -32601, f"method not found: {method}")


def serve(stdin=None, stdout=None) -> int:
    import warnings
    warnings.simplefilter("ignore")
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp = _error(None, -32700, f"parse error: {e}")
        else:
            resp = handle(msg) if isinstance(msg, dict) else _error(None, -32600, "invalid request")
        if resp is not None:
            stdout.write(json.dumps(resp) + "\n")
            stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(serve())
