"""Backward-compatible root entry point.

The consolidated CLI lives in pipeline.py.  Existing commands such as
`python main.py all` and `python main.py pca` remain supported.
"""

from pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())
