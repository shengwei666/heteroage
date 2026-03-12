import logging
import os
import sys

def setup_logger(output_dir=None, name="heteroage", level=logging.INFO):
    """
    [Utility]: Centralized Experiment Logger
    
    Configures a thread-safe logger with stream and file handlers. 
    Prevents duplicate handler attachment across repeated experimental runs.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 

    # Clean up existing handlers to prevent duplicate logging in interactive environments
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter: Includes timestamp, level, and module name for precise debugging
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Handler: Standard Output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 2. File Handler: Persistent storage for experimental audit
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = os.path.join(output_dir, "session.log")
        
        # Use 'a' (append) to preserve history across crashes, but starts fresh on new sessions
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
    return logger