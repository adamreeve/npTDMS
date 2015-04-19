import time


class Timer(object):
    def __init__(self, log, description):
        self._log = log
        self._description = description

    def __enter__(self):
        self._start_time = time.clock()

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = (time.clock() - self._start_time) * 1.0e3
        self._log.info("{0}: Took {1} ms".format(
            self._description, elapsed_time))
