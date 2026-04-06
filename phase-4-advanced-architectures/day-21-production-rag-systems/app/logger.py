"""
Centralised logging setup.
Writes structured logs to both console and a rotating file.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from app.config import LOG_PATH

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (5 MB max, keep 3 backups)
    fh = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
