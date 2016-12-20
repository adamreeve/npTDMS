"""Conversions to and from bytes representation of values in TDMS files"""

from datetime import datetime, timedelta
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


class TdmsValue(object):
    def __eq__(self, other):
        return self.bytes == other.bytes and self.value == other.value

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.value)


class Bytes(TdmsValue):
    def __init__(self, value):
        self.value = value
        self.bytes = value


class String(TdmsValue):
    enum_value = 0x20

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


class TimeStamp(TdmsValue):
    enum_value = 0x44

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
        data = file.read(data_type.length)
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


class Int32(StructValue):
    enum_value = 3
    size = 4
    struct_declaration = "l"


class Uint32(StructValue):
    enum_value = 7
    size = 4
    size = 4
    struct_declaration = "L"


class Int64(StructValue):
    enum_value = 4
    size = 8
    struct_declaration = "q"


class Uint64(StructValue):
    enum_value = 8
    size = 8
    struct_declaration = "Q"


class Boolean(StructValue):
    enum_value = 0x21
    size = 1
    struct_declaration = "b"


class SingleFloat(StructValue):
    enum_value = 9
    size = 4
    struct_declaration = "f"


class DoubleFloat(StructValue):
    enum_value = 10
    size = 8
    struct_declaration = "d"
