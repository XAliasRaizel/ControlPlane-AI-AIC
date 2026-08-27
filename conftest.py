# conftest.py — repo root
#
# This file's presence tells pytest that the repository root is the rootdir.
# pytest automatically inserts rootdir onto sys.path when it finds a conftest.py
# here, which makes `import backend` and `import ml` work under the bare
# `pytest` console script (no `-m` flag needed) — matching CI behaviour.
#
# Do not remove: without this, bare `pytest -q` on CI (ubuntu-latest) raises
#   ModuleNotFoundError: No module named 'backend'
# because the console-script entry-point does not add cwd to sys.path.
