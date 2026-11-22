"""OpenSSH configuration and keys; never place passwords on a command line."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass


@dataclass(frozen=True)
class SSH:
    host: str
    connect_timeout: int = 5

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", self.host):
            raise ValueError("Use a hostname or an alias from ~/.ssh/config")

    def options(self) -> list[str]:
        return [
            "ssh",
            "-oBatchMode=yes",
            "-oStrictHostKeyChecking=yes",
            f"-oConnectTimeout={self.connect_timeout}",
        ]

    def command(self, arguments: list[str]) -> list[str]:
        if not arguments:
            raise ValueError("Empty remote command")
        return [*self.options(), self.host, shlex.join(arguments)]

    def tunnel(self, local_port: int, remote_port: int) -> list[str]:
        if not all(
            isinstance(p, int) and not isinstance(p, bool) and 0 < p < 65536
            for p in (local_port, remote_port)
        ):
            raise ValueError("Invalid tunnel port")
        return [
            *self.options(),
            "-N",
            "-oExitOnForwardFailure=yes",
            "-L",
            f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
            self.host,
        ]
