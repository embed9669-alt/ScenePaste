#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`scenepaste.tools.split`."""
from scenepaste.tools.split import *  # noqa: F401,F403
from scenepaste.tools.split import main

if __name__ == "__main__":
    raise SystemExit(main())
