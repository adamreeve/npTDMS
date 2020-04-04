from functools import wraps
import logging
import time

try:
    from collections import OrderedDict
except ImportError:
    try:
        # ordereddict available on pypi for Python < 2.7
        from ordereddict import OrderedDict
    except ImportError:
        # Otherwise fall back on normal dict
        OrderedDict = dict


def cached_property(func):
    """ Wraps a method on a class to make it a property and caches the result the first time it is evaluated
    """
    attr_name = '_cached_prop_' + func.__name__

    @property
    @wraps(func)
    def get(self):
        try:
            return getattr(self, attr_name)
        except AttributeError:
            value = func(self)
            setattr(self, attr_name, value)
            return value

    return get


class Timer(object):
    """ Context manager for logging the  time taken  for an operation
    """

    def __init__(self, log, description):
        self._enabled = log.isEnabledFor(logging.INFO)
        self._log = log
        self._description = description
        self._start_time = None

    def __enter__(self):
        if not self._enabled:
            return self
        try:
            self._start_time = time.perf_counter()
        except AttributeError:
            # Python < 3.3
            self._start_time = time.clock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._enabled:
            return
        try:
            end_time = time.perf_counter()
        except AttributeError:
            # Python < 3.3
            end_time = time.clock()

        elapsed_time = (end_time - self._start_time) * 1.0e3
        self._log.info("{0}: Took {1} ms".format(self._description, elapsed_time))
