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


class TdmsType(object):
    size = None

    def __init__(self):
        self.value = None
        self.bytes = None

    def __eq__(self, other):
        return self.bytes == other.bytes and self.value == other.value

    def __repr__(self):
        if self.value is None:
            return "%s" % self.__class__.__name__
        return "%s(%r)" % (self.__class__.__name__, self.value)

    @classmethod
    def read(cls, file, endianness="<"):
        raise TypeError("Unsupported data type to read: %r" % cls)


class Bytes(TdmsType):
    def __init__(self, value):
        self.value = value
        self.bytes = value


class StructType(TdmsType):
    def __init__(self, value):
        self.value = value
        self.bytes = _struct_pack('<' + self.struct_declaration, value)

    @classmethod
    def read(cls, file, endianness="<"):
        bytes = file.read(cls.size)
        return _struct_unpack(endianness + cls.struct_declaration, bytes)[0]


@tds_data_type(0, None)
class Void(TdmsType):
    pass


@tds_data_type(1, np.int8)
class Int8(StructType):
    size = 1
    struct_declaration = "b"


@tds_data_type(2, np.int16)
class Int16(StructType):
    size = 2
    struct_declaration = "h"


@tds_data_type(3, np.int32)
class Int32(StructType):
    size = 4
    struct_declaration = "l"


@tds_data_type(4, np.int64)
class Int64(StructType):
    size = 8
    struct_declaration = "q"


@tds_data_type(5, np.uint8)
class Uint8(StructType):
    size = 1
    struct_declaration = "B"


@tds_data_type(6, np.uint16)
class Uint16(StructType):
    size = 2
    struct_declaration = "H"


@tds_data_type(7, np.uint32)
class Uint32(StructType):
    size = 4
    struct_declaration = "L"


@tds_data_type(8, np.uint64)
class Uint64(StructType):
    size = 8
    struct_declaration = "Q"


@tds_data_type(9, np.single)
class SingleFloat(StructType):
    size = 4
    struct_declaration = "f"


@tds_data_type(10, np.double)
class DoubleFloat(StructType):
    size = 8
    struct_declaration = "d"


@tds_data_type(11, None)
class ExtendedFloat(TdmsType):
    pass


@tds_data_type(12, None)
class DoubleFloatWithUnit(TdmsType):
    size = 8
    pass


@tds_data_type(13, None)
class ExtendedFloatWithUnit(TdmsType):
    pass


@tds_data_type(0x19, None)
class SingleFloatWithUnit(TdmsType):
    size = 4
    pass


@tds_data_type(0x20, None)
class String(TdmsType):
    def __init__(self, value):
        self.value = value
        content = value.encode('utf-8')
        length = _struct_pack('<L', len(content))
        self.bytes = length + content

    @staticmethod
    def read(file, endianness="<"):
        size = Uint32.read(file, endianness)
        return file.read(size).decode('utf-8')


@tds_data_type(0x21, np.bool8)
class Boolean(StructType):
    size = 1
    struct_declaration = "b"


@tds_data_type(0x44, None)
class TimeStamp(TdmsType):
    # Time stamps are stored as number of seconds since
    # 01/01/1904 00:00:00.00 UTC, ignoring leap seconds,
    # and number of 2^-64 fractions of a second.
    # Note that the TDMS epoch is not the Unix epoch.
    _tdms_epoch = datetime(1904, 1, 1, 0, 0, 0, tzinfo=timezone)
    _fractions_per_microsecond = float(10**-6) / 2**-64

    size = 16

    def __init__(self, value):
        if isinstance(value, np.datetime64):
            value = value.astype(datetime)
            # numpy converts a datetime with no time part to
            # datetime.date instead of datetime.datetime
            if not isinstance(value, datetime):
                value = datetime.combine(value, datetime.min.time())
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone)
        self.value = value
        epoch_delta = value - self._tdms_epoch
        seconds_per_day = 86400
        seconds = epoch_delta.days * seconds_per_day + epoch_delta.seconds
        second_fractions = int(
            epoch_delta.microseconds * self._fractions_per_microsecond)
        self.bytes = _struct_pack('<Qq', second_fractions, seconds)

    @classmethod
    def read(cls, file, endianness="<"):
        data = file.read(16)
        if endianness is "<":
            (second_fractions, seconds) = _struct_unpack(
                endianness + 'Qq', data)
        else:
            (seconds, second_fractions) = _struct_unpack(
                 endianness + 'Qq', data)
        micro_seconds = (
            float(second_fractions) / cls._fractions_per_microsecond)
        # Adding timedelta with seconds ignores leap
        # seconds, so this is correct
        return (cls._tdms_epoch + timedelta(seconds=seconds) +
                timedelta(microseconds=micro_seconds))


@tds_data_type(0xFFFFFFFF, np.int16)
class DaqMxRawData(TdmsType):
    size = 2
