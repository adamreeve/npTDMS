"""Conversions to and from bytes representation of values in TDMS files"""

from datetime import datetime, timedelta
import numpy as np
import struct
try:
    import pytz
except ImportError:
    pytz = None


_struct_pack = struct.pack
_struct_unpack = struct.unpack

if pytz:
    # Use UTC time zone if pytz is installed
    timezone = pytz.utc
else:
    timezone = None


tds_data_types = {}
numpy_data_types = {}


def tds_data_type(enum_value, np_type):
    def decorator(cls):
        cls.enum_value = enum_value
        cls.nptype = np_type
        if enum_value is not None:
            tds_data_types[enum_value] = cls
        if np_type is not None:
            numpy_data_types[np_type] = cls
        return cls
    return decorator


class TdmsValue(object):
    def __eq__(self, other):
        return self.bytes == other.bytes and self.value == other.value

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)


class Bytes(TdmsValue):
    def __init__(self, value):
        self.value = value
        self.bytes = value


@tds_data_type(0x20, None)
class String(TdmsValue):
    def __init__(self, value):
        self.value = value
        content = value.encode('utf-8')
        length = _struct_pack('<L', len(content))
        self.bytes = length + content

    @staticmethod
    def read(file):
        size_bytes = file.read(4)
        size = _struct_unpack("<L", size_bytes)[0]
        return file.read(size).decode('utf-8')


@tds_data_type(0x44, None)
class TimeStamp(TdmsValue):
    # Time stamps are stored as number of seconds since
    # 01/01/1904 00:00:00.00 UTC, ignoring leap seconds,
    # and number of 2^-64 fractions of a second.
    # Note that the TDMS epoch is not the Unix epoch.
    _tdms_epoch = datetime(1904, 1, 1, 0, 0, 0, tzinfo=timezone)
    _fractions_per_microsecond = float(10**-6) / 2**-64

    def __init__(self, value):
        self.value = value
        epoch_delta = value - self._tdms_epoch
        seconds_per_day = 86400
        seconds = epoch_delta.days * seconds_per_day + epoch_delta.seconds
        second_fractions = int(
            epoch_delta.microseconds * self._fractions_per_microsecond)
        self.bytes = _struct_pack('<Qq', second_fractions, seconds)

    @staticmethod
    def read(file):
        data = file.read(16)
        (second_fractions, seconds) = _struct_unpack('<Qq', data)
        micro_seconds = (
            float(second_fractions) / self._fractions_per_microsecond)
        # Adding timedelta with seconds ignores leap
        # seconds, so this is correct
        return (self._tdms_epoch + timedelta(seconds=seconds)
                + timedelta(microseconds=micro_seconds))


class StructValue(TdmsValue):
    def __init__(self, value):
        self.value = value
        self.bytes = _struct_pack("<" + self.struct_declaration, value)

    @staticmethod
    def read(file):
        bytes = file.read(self.size)
        return _struct_unpack("<" + self.struct_declaration, bytes)[0]


@tds_data_type(3, np.int32)
class Int32(StructValue):
    size = 4
    struct_declaration = "l"


@tds_data_type(7, np.uint32)
class Uint32(StructValue):
    size = 4
    struct_declaration = "L"


@tds_data_type(4, np.int64)
class Int64(StructValue):
    size = 8
    struct_declaration = "q"


@tds_data_type(8, np.uint64)
class Uint64(StructValue):
    size = 8
    struct_declaration = "Q"


@tds_data_type(0x21, np.bool8)
class Boolean(StructValue):
    size = 1
    struct_declaration = "b"


@tds_data_type(9, np.single)
class SingleFloat(StructValue):
    size = 4
    struct_declaration = "f"


@tds_data_type(10, np.double)
class DoubleFloat(StructValue):
    size = 8
    struct_declaration = "d"
