#!/usr/bin/env python3
"""Small dependency-free TCP relay used when the pinned image lacks socat."""

from __future__ import annotations

import socket
import socketserver
import sys
import threading


def _pump(source: socket.socket, destination: socket.socket) -> None:
    try:
        while data := source.recv(65536):
            destination.sendall(data)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class RelayHandler(socketserver.BaseRequestHandler):
    target_host: str
    target_port: int

    def handle(self) -> None:
        with socket.create_connection((self.target_host, self.target_port)) as upstream:
            reverse = threading.Thread(
                target=_pump,
                args=(upstream, self.request),
                daemon=True,
            )
            reverse.start()
            _pump(self.request, upstream)
            reverse.join()


class RelayServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: tcp_relay.py LISTEN_HOST LISTEN_PORT TARGET_HOST TARGET_PORT"
        )
    listen_host, listen_port, target_host, target_port = sys.argv[1:]
    handler = type(
        "ConfiguredRelayHandler",
        (RelayHandler,),
        {"target_host": target_host, "target_port": int(target_port)},
    )
    with RelayServer((listen_host, int(listen_port)), handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
