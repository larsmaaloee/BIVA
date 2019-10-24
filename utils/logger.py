import logging
import sys

def init_logging(path):
    logger = logging.getLogger('model_logger')
    for hdlr in logger.handlers: logger.removeHandler(hdlr)
    hdlr = logging.FileHandler(path)
    formatter = logging.Formatter('%(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    return logger