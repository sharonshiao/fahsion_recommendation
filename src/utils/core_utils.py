import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(level=logging.DEBUG, log_file="tmp.log", remove_existing=False):

    # Configure the logger
    root_logger = logging.getLogger()  # Get the root logger
    root_logger.setLevel(level)  # Set the root logger to debug

    # Remove all existing handlers (to prevent duplicate logging)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    if remove_existing:
        logger.info(f"Removing existing log file: {log_file}")
        if os.path.exists(log_file):
            os.remove(log_file)

    # Create console handler
    logger.info(f"Creating console handler with level: {level}")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)  # Console handler also listens to DEBUG level

    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_str)
    console_handler.setFormatter(formatter)

    # Add console handler to the logger
    logger.debug("Adding console handler to the logger")
    root_logger.addHandler(console_handler)

    # Add FileHandler
    logger.info(f"Creating file handler with level: {level}")
    fhandler = logging.FileHandler(filename=log_file, mode="a")
    fhandler.setFormatter(formatter)
    root_logger.addHandler(fhandler)
    logger.debug(f"Logging setup complete to {log_file}")

    return root_logger


# # FIX, only for testing
# def setup_logging(
#     root_logger=None, level=logging.INFO, format_str=None, log_file=None, console_output=True, file_mode="a"
# ):
#     if not root_logger:
#         root_logger = logging.getLogger()
#         root_logger.setLevel(level)

#     # Remove all existing handlers (to prevent duplicate logging)
#     if root_logger.hasHandlers():
#         root_logger.handlers.clear()

#     # Create console handler
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(level)  # Console handler also listens to DEBUG level

#     if not format_str:
#         format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#     formatter = logging.Formatter(format_str)
#     console_handler.setFormatter(formatter)

#     # Add console handler to the logger
#     if console_output:
#         root_logger.addHandler(console_handler)

#     # Add FileHandler
#     fhandler = logging.FileHandler(filename=log_file, mode=file_mode)
#     fhandler.setFormatter(formatter)
#     if log_file is not None:
#         root_logger.addHandler(fhandler)
#     return root_logger


# FIXME: To be removed later after the new function is tested
# def _setup_logging(level=logging.INFO, format_str=None, log_file=None, console_output=True, file_mode="a"):
#     """
#     Configure logging with consistent format across the project.

#     Args:
#         level: Logging level (default: INFO)
#         format_str: Custom format string (optional)
#         log_file: Path to log file (optional)
#         console_output: Whether to output logs to console (default: True)
#         file_mode: File mode for log file ('a' for append, 'w' for write) (default: 'a')
#     """
#     if format_str is None:
#         format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#     # Ensure all child loggers propagate to root
#     for name in logging.root.manager.loggerDict:
#         logger = logging.getLogger(name)
#         logger.propagate = True
#         logger.handlers = []  # Remove any existing handlers from child loggers

#     # Create formatter
#     formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")

#     # Configure root logger
#     root_logger = logging.getLogger()
#     root_logger.setLevel(level)

#     # Remove existing handlers to avoid duplicates
#     for handler in root_logger.handlers[:]:
#         root_logger.removeHandler(handler)

#     # Add console handler if requested
#     if console_output:
#         print("Adding console handler")
#         logger.info("Adding console handler")
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         root_logger.addHandler(console_handler)

#     # Add file handler if specified
#     if log_file:
#         print(f"Adding file handler to {log_file} with mode {file_mode}")
#         logger.info(f"Adding file handler to {log_file} with mode {file_mode}")
#         file_handler = logging.FileHandler(log_file, mode=file_mode)
#         file_handler.setFormatter(formatter)
#         root_logger.addHandler(file_handler)

#     # Return the root logger
#     return root_logger


def set_seed(seed=42):
    """
    Set the seed for random number generation to ensure reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.mps.manual_seed(seed)


def human_readable_size(size_bytes):
    """
    Convert bytes to human readable format (KB, MB, GB, TB)

    Args:
        size_bytes (int): Size in bytes

    Returns:
        str: Human readable string with appropriate unit
    """
    if size_bytes < 0:
        raise ValueError("Size must be non-negative")

    units = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1

    if unit_index == 0:  # Still in bytes
        return f"{size:.0f} {units[unit_index]}"
    else:
        return f"{size:.2f} {units[unit_index]}"


def convert_dict_values_to_float(d: dict) -> dict:
    """
    Convert all values in a dictionary to float and handle
    """
    for k, v in d.items():
        if isinstance(v, str):
            print(f"Converting {v} to float")
            try:
                d[k] = float(v)
            except ValueError:
                pass
        elif isinstance(v, dict):
            print(f"Converting {v} to float")
            d[k] = convert_dict_values_to_float(v)
    return d


def decode_lightgbm_params(params: dict) -> dict:
    """
    Decode the lightgbm parameters to handle integer/floats from json.
    """
    cols_float = ["reg_lambda", "learning_rate", "colsample_bytree", "reg_alpha", "subsample"]
    cols_int = [
        "random_state",
        "n_estimators",
        "verbosity",
        "min_child_samples",
        "num_leaves",
        "subsample_freq",
    ]
    cols_bool = ["feature_pre_filter"]
    cols_float = set(cols_float)
    cols_int = set(cols_int)
    cols_bool = set(cols_bool)

    for k, v in params.items():
        if k in cols_float:
            params[k] = float(v)
        elif k in cols_int:
            params[k] = int(v)
        elif k in cols_bool:
            params[k] = bool(v)
    return params
