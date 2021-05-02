from configs.base_config import *
from utils.datetime_utils import get_dt_string
import logging
import logging.handlers
import os

    
def initiate_logger():
    log_file_name = base_dir + 'logs/' + get_dt_string() + ".log"
    handler = logging.handlers.WatchedFileHandler(log_file_name)
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root.addHandler(handler)