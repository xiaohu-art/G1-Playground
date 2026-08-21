import logging
from pathlib import Path

import colorlog


def setup_logger(name: str = "g1_playground") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.disabled = False
    for existing_name, existing in logging.root.manager.loggerDict.items():
        if isinstance(existing, logging.Logger) and existing_name.startswith(f"{name}."):
            existing.disabled = False
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    color_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s", "%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger
