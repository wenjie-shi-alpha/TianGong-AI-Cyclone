"""Exceptions used across the initial tracker workflow."""


class NoEyeException(Exception):
    """Raised when no cyclone eye can be detected during tracking."""


__all__ = ["NoEyeException"]
