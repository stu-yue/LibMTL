import os
import random
from collections import OrderedDict
from datetime import datetime
from enum import Enum
from logging import getLogger
import logging.config
import pdb


def get_local_time():
    cur_time = datetime.now().strftime("%b-%d-%Y-%H-%M-%S")
    return cur_time

def create_dirs(dir_paths):
    """
    :param dir_paths:
    :return:
    """
    if not isinstance(dir_paths, (list, tuple)):
        dir_paths = [dir_paths]
    for dir_path in dir_paths:
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

str_level_dict = {
    "notest": logging.NOTSET,
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

def init_logger_config(
        log_level, result_root, is_train=True
):
    if log_level not in str_level_dict:
        raise KeyError

    level = str_level_dict[log_level]
    file_name = "{}-{}.log".format(
        "train" if is_train else "test", get_local_time()
    )
    log_path = os.path.join(result_root, file_name)
    logging_config = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "simple": {
                "format": "%(asctime)s [%(levelname)s]: %(message)s"
            }
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": level,
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "simple",
                "filename": log_path,
                "maxBytes": 100 * 1024 * 1024,
                "backupCount": 3,
            },
        },
        "loggers": {
            "": {
                "handlers": [
                    "console",
                    "file",
                ],
                "level": level,
                "propagate": True,
            }
        },
    }
    logging.config.dictConfig(logging_config)
