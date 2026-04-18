"""Meta Watcher package."""

import os as _os

# OpenCV's Orbbec (obsensor) and generic V4L2 backends print a flurry of
# `can't open camera by index` warnings to stderr as they probe the system
# during `import cv2`. These are noise on every machine we support, and
# suppressing them keeps the CLI output legible. Set env vars *before* cv2
# is ever imported transitively.
_os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
_os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_OBSENSOR", "0")

__all__ = ["__version__"]

__version__ = "0.1.0"
