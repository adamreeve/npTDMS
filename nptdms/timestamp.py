from datetime import datetime, timedelta
import numpy as np


EPOCH = np.datetime64('1904-01-01 00:00:00', 's')


class TdmsTimestamp(object):
    """ A Timestamp from a TDMS file

        The TDMS format stores timestamps as a signed number of seconds since the epoch 1904-01-01 00:00:00 UTC
        and number of positive fractions (2^-64) of a second.

        :ivar ~.seconds: Seconds since the epoch as a signed integer
        :ivar ~.second_fractions: A positive number of 2^-64 fractions of a second
    """

    def __init__(self, seconds, second_fractions):
        self.seconds = seconds
        self.second_fractions = second_fractions

    def __repr__(self):
        return "TdmsTimestamp({0}, {1})".format(self.seconds, self.second_fractions)

    def __str__(self):
        dt = EPOCH + np.timedelta64(self.seconds, 's')
        fraction_string = "{0:.6f}".format(self.second_fractions * 2.0 ** -64).split('.')[1]
        return "{0}.{1}".format(dt, fraction_string)

    def as_datetime64(self, resolution='us'):
        """ Convert this timestamp to a numpy datetime64 object

            :param resolution: The resolution of the datetime64 object to create as a numpy unit code.
                Must be one of 's', 'ms', 'us', 'ns' or 'ps'
        """
        try:
            fractions_per_step = _fractions_per_step[resolution]
        except KeyError:
            raise ValueError("Unsupported resolution for converting to numpy datetime64: '{0}'".format(resolution))
        return (
                EPOCH +
                np.timedelta64(self.seconds, 's') +
                ((self.second_fractions / fractions_per_step) * np.timedelta64(1, resolution)))

    def as_datetime(self):
        """ Convert this timestamp to a Python datetime.datetime object
        """
        fractions_per_us = _fractions_per_step['us']
        microseconds = (self.second_fractions / fractions_per_us)
        return datetime(1904, 1, 1, 0, 0, 0) + timedelta(seconds=self.seconds) + timedelta(microseconds=microseconds)


class TimestampArray(np.ndarray):
    """ A numpy array of TDMS timestamps

        Indexing into a TimestampArray returns TdmsTimestamp objects.
    """

    def __new__(cls, input_array):
        """ Create a new TimestampArray

            The input array must be a structured numpy array with 'seconds' and 'second_fractions' fields.
        """
        obj = np.asarray(input_array).view(cls)
        field_names = input_array.dtype.names
        if field_names == ('second_fractions', 'seconds'):
            obj._field_indices = (1, 0)
        elif field_names == ('seconds', 'second_fractions'):
            obj._field_indices = (0, 1)
        else:
            raise ValueError("Input array must have a dtype with 'seconds' and 'second_fractions' fields")
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._field_indices = getattr(obj, '_field_indices', '<')

    def __getitem__(self, item):
        val = super(TimestampArray, self).__getitem__(item)
        if isinstance(item, str):
            # Getting a field, we don't want to return this as a TimestampArray
            # but as a normal numpy ndarray
            return val.view(np.ndarray)
        if isinstance(item, (int, np.number)):
            # Getting a single item
            return TdmsTimestamp(val[self._field_indices[0]], val[self._field_indices[1]])
        # else getting a slice returns a new TimestampArray
        return val

    @property
    def seconds(self):
        """ The number of seconds since the TDMS epoch (1904-01-01 00:00:00 UTC) as a numpy array
        """
        return self['seconds']

    @property
    def second_fractions(self):
        """ The number of 2**-64 fractions of a second as a numpy array
        """
        return self['second_fractions']

    def as_datetime64(self, resolution='us'):
        """ Convert to an array of numpy datetime64 objects

            :param resolution: The resolution of the datetime64 objects to create as a numpy unit code.
                Must be one of 's', 'ms', 'us', 'ns' or 'ps'
        """
        try:
            fractions_per_step = _fractions_per_step[resolution]
        except KeyError:
            raise ValueError("Unsupported resolution for converting to numpy datetime64: '{0}'".format(resolution))
        return (
                EPOCH +
                self['seconds'] * np.timedelta64(1, 's') +
                (self['second_fractions'] / fractions_per_step) * np.timedelta64(1, resolution))


_fractions_per_step = {
    's': 1.0 / 2 ** -64,
    'ms': (10 ** -3) / 2 ** -64,
    'us': (10 ** -6) / 2 ** -64,
    'ns': (10 ** -9) / 2 ** -64,
    'ps': (10 ** -12) / 2 ** -64,
}
