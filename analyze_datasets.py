#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`scenepaste.tools.analyze`."""
from scenepaste.tools.analyze import *  # noqa: F401,F403
from scenepaste.tools.analyze import main

if __name__ == "__main__":
    raise SystemExit(main())
