"""
logging_config.py
=================

Central logging setup for the pdf_pipeline package. Call ``setup_pipeline_logging``
once at process start (e.g. from ``example_usage.py``) to enable step-by-step
logs from pipeline, layout, OCR, and table extraction modules.
"""

from __future__ import annotations

import logging
from typing import Optional, Union


def setup_pipeline_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logging for the pipeline.

    Parameters
    ----------
    level:
        Logging level (``logging.DEBUG``, ``logging.INFO``, etc.) or a level name
        such as ``"DEBUG"``.
    log_file:
        Optional path to also write logs to a file.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)

    # Keep noisy third-party libraries quieter unless DEBUG is requested.
    if level > logging.DEBUG:
        for name in ("PIL", "onnxruntime", "urllib3"):
            logging.getLogger(name).setLevel(logging.WARNING)
