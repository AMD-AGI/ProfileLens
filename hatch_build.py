# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Hatch build metadata for ProfileLens's bundled Linux tool library."""

from __future__ import annotations

from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    """Prevent the bundled ELF library from being labeled cross-platform."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        build_data["tag"] = "py3-none-linux_x86_64"
        build_data["pure_python"] = False
