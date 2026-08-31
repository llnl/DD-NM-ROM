import logging
import logging.config

from . import config as cfg

_DEFAULT_LOG_LEVEL = logging.WARNING
#_DEFAULT_LOG_LEVEL = logging.DEBUG

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        # "file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "level": "INFO",
        #     "formatter": "detailed",
        #     "filename": "app.log",
        #     "maxBytes": 10485760,  # 10MB
        #     "backupCount": 5,
        # },
    },
    "loggers": {
        # Package-level configuration
        "dd_nm_rom": {
            #"handlers": ["console", "file"],
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
    # "root": {
    #     "handlers": ["console"],
    #     "level": "WARNING",
    # },
}

def _setup_logger(name, log_level=None):
  """
  Setup package wide logger
  If log_level is None, then this will use the default / environment value
  The explicit log_level option is mostly used to override or reinit logging from a function
  """
  verbose = cfg.update_from_env("DDNMROM_VERBOSE")

  if log_level is None:
    log_level = cfg.update_from_env("DDNMROM_LOG_LEVEL", _DEFAULT_LOG_LEVEL)

  if verbose > 0:
    #LOGGING_CONFIG["root"]["level"] = _DEFAULT_LOG_LEVEL
    LOGGING_CONFIG["loggers"]["dd_nm_rom"]["level"] = log_level
  else:
    #LOGGING_CONFIG["root"]["handlers"] = []
    LOGGING_CONFIG["loggers"]["dd_nm_rom"]["handlers"] = []

  # Apply dictionary schema configuration directly at initialization
  logging.config.dictConfig(LOGGING_CONFIG)

  _log = logging.getLogger(name)

  if verbose == 0:
    _log.handlers.clear()
    _log.addHandler(logging.NullHandler())

  return _log
