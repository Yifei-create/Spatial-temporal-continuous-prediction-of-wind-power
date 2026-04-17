import logging
import sys
import os
import os.path as osp

def get_logger(name="STCWPF", log_dir=None, log_file=None):
    """Get logger instance"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_dir and log_file:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(osp.join(log_dir, log_file), mode="w")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
