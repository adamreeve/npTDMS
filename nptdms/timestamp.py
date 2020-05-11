import numpy as np


EPOCH = np.datetime64('1904-01-01 00:00:00', 's')


class TdmsTimestamp(object):
    """ A Timestamp from a TDMS file

        The TDMS format stores timestamps as a signed number of seconds since the epoch 1904-01-01 00:00:00 UTC
        and number of positive fractions (2^-64) of a second.

        :ivar ~.seconds: Seconds since the epoch as a signed integer
        :ivar ~.second_fractions: A positive number of 2^-64 fractions of a second
    """

    _fractions_per_step = {
        's': 1.0 / 2 ** -64,
        'ms': (10 ** -3) / 2 ** -64,
        'us': (10 ** -6) / 2 ** -64,
        'ns': (10 ** -9) / 2 ** -64,
        'ps': (10 ** -12) / 2 ** -64,
    }

    def __init__(self, seconds, second_fractions):
        self.seconds = seconds
        self.second_fractions = second_fractions

    def as_datetime64(self, resolution='us'):
        try:
            fractions_per_step = self._fractions_per_step[resolution]
        except KeyError:
            raise ValueError("Unsupported resolution for converting to numpy datetime64: {0}".format(resolution))
        return (
                EPOCH +
                np.timedelta64(self.seconds, 's') +
                ((self.second_fractions / fractions_per_step) * np.timedelta64(1, resolution)))
