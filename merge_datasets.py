#!/usr/bin/env python3
"""Compatibility wrapper for :mod:`scenepaste.tools.merge`."""
from scenepaste.tools.merge import *  # noqa: F401,F403
from scenepaste.tools.merge import main

if __name__ == "__main__":
    raise SystemExit(main())
