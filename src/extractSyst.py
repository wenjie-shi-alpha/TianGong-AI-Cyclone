#!/usr/bin/env python3
"""CLI shim that delegates to the refactored environment extractor package."""

from __future__ import annotations

import sys

from environment_extractor import main as run_main
from environment_extractor.deps import ERROR_MESSAGE, MissingDependencyError


def main() -> None:
    try:
        run_main()
    except MissingDependencyError:
        print(ERROR_MESSAGE)
        sys.exit(1)


if __name__ == "__main__":
    main()
