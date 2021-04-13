"""
@PKG: EdgeNNGenerator
@DESC: this is the Logger Settings for EdgeNNGenerator.
@DATE: Year 2020
"""

import os
import logging
import logging.handlers
'''
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - levelName:%(levelname)s in %(module)s fileName:%(filename)s funcName:%(funcName)s lineNumber:%(lineno)d threadName:%(threadName)s outputMessage:%(message)s",
    datefmt='[%d/%b/%Y %H:%M:%S]',
    filename='./logs/logging.log'
)
'''


def LoggerSetup(logFile='./logs/loggingMsg.log', fileLevel='INFO', streamLevel='DEBUG', recordOpt=False, maxMB=None, backupCnt=3):

    logger = logging.getLogger()
    logger.setLevel('DEBUG')  # the lowest level
    BASIC_FORMAT = ("•".join(["%(asctime)s %(levelname)s", "%(filename)s(%(lineno)d)%(funcName)s:〔%(message)s〕"]))
    SIMPLE_FORMAT = (" ".join(["%(asctime)s %(levelname)s", "%(message)s"]))
    DATE_FORMAT = "%Y/%b/%d %H:%M:%S"

    S_formatter = logging.Formatter(SIMPLE_FORMAT, DATE_FORMAT)

    chlr = logging.StreamHandler()  # stream handler
    chlr.setFormatter(S_formatter)  # set stream format
    chlr.setLevel(streamLevel)  # default using logger level

    logger.addHandler(chlr)

    if recordOpt:
        F_formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

        if not os.path.exists(logFile):
            file_path = os.path.dirname(logFile)
            os.makedirs(file_path, exist_ok=True)
            os.mknod(logFile)
            logger.info("creating logging file: {}".format(logFile))
        else:
            logger.info("logging file exists: {}".format(logFile))

        if maxMB is None:
            fhlr = logging.FileHandler(logFile)  # file handler
        else:
            fhlr = logging.handlers.RotatingFileHandler(logFile, mode='a', maxBytes=maxMB * 1024 * 1024,
                                                        backupCount=backupCnt, encoding=None, delay=0)
        fhlr.setFormatter(F_formatter)  # set file format
        fhlr.setLevel(fileLevel)  # default using logger level

        logger.addHandler(fhlr)

    return logger


def print_info(info_string, info_title, splitter, spl_length, seperator, logger=None):
    if spl_length is None:
        spl_length = int(max(1, int((len(info_string) - len(info_title)) / 2)))
    else:
        spl_length = int(max(1, round(spl_length, 0)))
    if logger:
        logger.info(splitter * spl_length + seperator + info_title + seperator + splitter * spl_length)
        logger.info(seperator + info_string)
        logger.info(splitter * (2 * spl_length + len(info_title) + 2))
    else:
        print(splitter * spl_length, info_title, splitter * spl_length, sep=seperator)
        print("", info_string, sep=seperator)
        print(splitter * spl_length, splitter * len(info_title), splitter * spl_length, sep=splitter)


def standard_output(output_string, level='debug', logger=None):
    if logger:
        if level == 'debug':
            logger.debug(output_string)
        elif level == 'info':
            logger.info(output_string)
        elif level == 'warning':
            logger.warning(output_string)
        elif level == 'error':
            logger.error(output_string)
        elif level == 'critical':
            logger.critical(output_string)
    else:
        print(output_string)


if __name__ == '__main__':
    logger = LoggerSetup()
    logger.warning("Test")
    print_info("All potential GPU devices are visible. None of them is masked.", "GPU SETTINGS", "—", 25, "|", logger)
    print_info("All potential GPU devices are visible. None of them is masked.", "GPU SETTINGS", "—", 25, "|", None)
    standard_output("test", 'critical', logger)
