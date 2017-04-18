import logging
import numpy as np


def describe(df):
    txt = ''
    for col in df:
        txt += col + '\n'
        txt += str(df[col].describe())
        if len(df[col].unique()) < 10:
            txt += str(np.unique(df[col].values, return_counts=True))
        txt += '\n'
    return txt


def make_logger(name=''):
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
