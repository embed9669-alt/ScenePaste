# Security Policy

## Supported versions

| Version        | Supported |
|----------------|-----------|
| 0.5.x (current)| Yes       |
| < 0.5          | Best effort only |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Email the maintainer listed on the GitHub repository profile, or open a private security advisory on GitHub if available. Include:

- ScenePaste version (`scenepaste --version`)
- affected component (CLI / Qt GUI / file parsers)
- steps to reproduce
- impact assessment (data overwrite, path traversal, model/code execution, etc.)

You should receive an acknowledgement within a few days when the maintainer is available.

## Scope notes

ScenePaste is a **local-first** tool. It reads images/JSON from user-chosen directories and writes datasets to disk. Treat untrusted LabelMe/JSON inputs and unexpected file paths carefully. Optional ML backends (`rembg`, future SAM) pull third-party code and models — review those dependencies separately.
