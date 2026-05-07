"""CLI entry-point shims for the hierarchy package.

Each module in this directory is a thin wrapper that imports `main` from the
corresponding implementation module and invokes it. Kept separate from the
implementation modules so that they can be run as scripts without polluting
the package's public namespace.
"""
