import logging


class LogManager(object):
    """ Manages multiple loggers from different modules
    """

    def __init__(self):
        self.log_level = logging.WARNING

        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.log_level)

        self.formatter = logging.Formatter(
            '[%(name)s %(levelname)s] %(message)s')
        self.console_handler.setFormatter(self.formatter)

        self.loggers = {}

    def get_logger(self, module_name):
        """ Return a logger for a module
        """
        log = logging.getLogger(module_name)
        log.setLevel(self.log_level)
        log.addHandler(self.console_handler)
        self.loggers[module_name] = log
        return log

    def set_level(self, level):
        """ Set the log level for all loggers that have been created
        """
        self.log_level = level
        self.console_handler.setLevel(level)
        for log in self.loggers.values():
            log.setLevel(level)


log_manager = LogManager()
