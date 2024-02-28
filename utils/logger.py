import logging
from logging import StreamHandler
from logging.handlers import TimedRotatingFileHandler
import os


logger_name = ''
log_initialize = False


def init_logger(cfg=None, name="default", filename="", loglevel="debug"):
    # LOG FORMATTING
    # https://docs.python.org/ko/3.8/library/logging.html#logrecord-attributes
    global logger_name
    global log_initialize
    if log_initialize is True:
        return

    log_format = "[%(asctime)s]-[%(levelname)s]-[%(name)s]-[%(module)s](%(process)d): %(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'

    if cfg is not None:
        # LOGGER NAME
        logger_name = cfg.logger_name if cfg.logger_name else logger_name
        _logger = logging.getLogger(logger_name)

        # LOG LEVEL
        log_level = cfg.log_level.upper() if cfg.log_level else loglevel.upper()
        _logger.setLevel(log_level)

        # LOG CONSOLE
        if cfg.console_log is True:
            _handler = StreamHandler()
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

        # LOG FILE
        if cfg.file_log is True:
            filename = os.path.join(cfg.file_log_dir, cfg.logger_name + '.log')
            logdir = os.path.dirname(filename)
            if not os.path.exists(logdir):
                os.makedirs(logdir)

            when = cfg.file_log_rotate_time if hasattr(cfg, "file_log_rotate_time") else "D"
            interval = cfg.file_log_rotate_interval if hasattr(cfg, "file_log_rotate_interval") else 1
            _handler = TimedRotatingFileHandler(
                filename=filename,
                when=when,
                interval=interval,
                backupCount=cfg.file_log_counter,
                encoding='utf8'
            )
            _handler.setLevel(log_level)

            # logger formatting
            formatter = logging.Formatter(log_format)
            _handler.setFormatter(formatter)
            _logger.addHandler(_handler)

    else:       # cfg is None
        logger_name = __name__
        log_level = loglevel.upper()
        _logger = logging.getLogger(logger_name)
        _logger.setLevel(log_level)

        # CONSOLE LOGGER
        _handler = StreamHandler()
        _handler.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

        # FILE LOGGER
        filename = os.path.join('./log', logger_name + '.log')
        logdir = os.path.dirname(filename)
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        _handler = TimedRotatingFileHandler(
            filename=filename,
            when="D",
            interval=1,
            backupCount=10,
            encoding='utf8'
        )
        _handler.setLevel(log_level)
        formatter = logging.Formatter(log_format)
        _handler.setFormatter(formatter)
        _logger.addHandler(_handler)

    _logger.info("Start Main logger")
    log_initialize = True


def get_logger():
    return logging.getLogger(logger_name)
