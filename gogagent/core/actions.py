"""Bounded graph-construction actions shared by every domain adapter."""

from __future__ import annotations

from enum import Enum


class MacroAction(str, Enum):
    """Typed graph grammar productions exposed to the policy."""

    ATTACH_ANALYST = "ATTACH_ANALYST"
    ATTACH_CHECKER = "ATTACH_CHECKER"
    ATTACH_REVISER = "ATTACH_REVISER"
    ATTACH_ALTERNATIVE = "ATTACH_ALTERNATIVE"
    STOP = "STOP"
