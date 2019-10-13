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


class Timer(object):
    """ Context manager for logging the  time taken  for an operation
    """

    def __init__(self, log, description):
        self._log = log
        self._description = description
        self._start_time = None

    def __enter__(self):
        try:
            self._start_time = time.perf_counter()
        except AttributeError:
            # Python < 3.3
            self._start_time = time.clock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            end_time = time.perf_counter()
        except AttributeError:
            # Python < 3.3
            end_time = time.clock()

        elapsed_time = (end_time - self._start_time) * 1.0e3
        self._log.info("{0}: Took {1} ms".format(
            self._description, elapsed_time))
