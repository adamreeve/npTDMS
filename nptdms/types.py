"""Conversions to and from bytes representation of values in TDMS files"""

import numpy as np
import struct
from nptdms.timestamp import TdmsTimestamp, TimestampArray


__all__ = [
    'numpy_data_types',
    'tds_data_types',
    'TdmsType',
    'Bytes',
    'Void',
    'Int8',
    'Int16',
    'Int32',
    'Int64',
    'Uint8',
    'Uint16',
    'Uint32',
    'Uint64',
    'SingleFloat',
    'DoubleFloat',
    'ExtendedFloat',
    'SingleFloatWithUnit',
    'DoubleFloatWithUnit',
    'ExtendedFloatWithUnit',
    'String',
    'Boolean',
    'TimeStamp',
    'ComplexSingleFloat',
    'ComplexDoubleFloat',
    'DaqMxRawData',
]


_struct_pack = struct.pack
_struct_unpack = struct.unpack


tds_data_types = {}
numpy_data_types = {}


def tds_data_type(enum_value, np_type, set_np_type=True):
    def decorator(cls):
        cls.enum_value = enum_value
        cls.nptype = None if np_type is None else np.dtype(np_type)
        if enum_value is not None:
            tds_data_types[enum_value] = cls
        if set_np_type and np_type is not None:
            numpy_data_types[np.dtype(np_type)] = cls
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
        raise NotImplementedError("Unsupported data type to read: %r" % cls)

    @classmethod
    def read_values(cls, file, number_values, endianness="<"):
        raise NotImplementedError("Unsupported data type to read: %r" % cls)


class Bytes(TdmsType):
    def __init__(self, value):
        self.value = value
        self.bytes = value


class StructType(TdmsType):
    struct_declaration = None
    nptype = None

    def __init__(self, value):
        self.value = value
        self.bytes = _struct_pack('<' + self.struct_declaration, value)

    @classmethod
    def read(cls, file, endianness="<"):
        read_bytes = file.read(cls.size)
        return _struct_unpack(endianness + cls.struct_declaration, read_bytes)[0]

    @classmethod
    def from_bytes(cls, byte_array, endianness="<"):
        """ Convert an array of bytes into a numpy array of data
        """
        array = byte_array.view()
        array.dtype = cls.nptype.newbyteorder(endianness)
        return array


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


@tds_data_type(0x19, np.single, set_np_type=False)
class SingleFloatWithUnit(StructType):
    size = 4
    struct_declaration = "f"


@tds_data_type(0x1A, np.double, set_np_type=False)
class DoubleFloatWithUnit(StructType):
    size = 8
    struct_declaration = "d"


@tds_data_type(0x1B, None)
class ExtendedFloatWithUnit(TdmsType):
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
        size_bytes = file.read(4)
        size = _struct_unpack(endianness + 'L', size_bytes)[0]
        return file.read(size).decode('utf-8')

    @classmethod
    def read_values(cls, file, number_values, endianness="<"):
        """ Read string raw data

            This is stored as an array of offsets
            followed by the contiguous string data.
        """
        offsets = [0]
        for i in range(number_values):
            offsets.append(Uint32.read(file, endianness))
        strings = []
        for i in range(number_values):
            s = file.read(offsets[i + 1] - offsets[i])
            strings.append(s.decode('utf-8'))
        return strings


@tds_data_type(0x21, np.bool8)
class Boolean(StructType):
    size = 1
    struct_declaration = "b"

    @classmethod
    def read(cls, file, endianness="<"):
        return bool(super(Boolean, cls).read(file, endianness))


@tds_data_type(0x44, None)
class TimeStamp(TdmsType):
    # Time stamps are stored as number of seconds since
    # 01/01/1904 00:00:00.00 UTC, ignoring leap seconds,
    # and number of 2^-64 fractions of a second.
    # Note that the TDMS epoch is not the Unix epoch.
    _tdms_epoch = np.datetime64('1904-01-01 00:00:00', 'us')
    _fractions_per_microsecond = float(10**-6) / 2**-64

    size = 16

    def __init__(self, value):
        if not isinstance(value, np.datetime64):
            value = np.datetime64(value, 'us')
        self.value = value
        epoch_delta = value - self._tdms_epoch

        seconds = int(epoch_delta / np.timedelta64(1, 's'))
        remainder = epoch_delta - np.timedelta64(seconds, 's')
        zero_delta = np.timedelta64(0, 's')
        if remainder < zero_delta:
            remainder = np.timedelta64(1, 's') + remainder
            seconds = seconds - 1
        microseconds = int(remainder / np.timedelta64(1, 'us'))
        second_fractions = int(microseconds * self._fractions_per_microsecond)
        self.bytes = _struct_pack('<Qq', second_fractions, seconds)

    @classmethod
    def read(cls, file, endianness="<"):
        data = file.read(16)
        if endianness == "<":
            (second_fractions, seconds) = _struct_unpack(
                endianness + 'Qq', data)
        else:
            (seconds, second_fractions) = _struct_unpack(
                 endianness + 'qQ', data)
        return TdmsTimestamp(seconds, second_fractions)

    @classmethod
    def from_bytes(cls, byte_array, endianness="<"):
        """ Convert an array of bytes to an array of timestamps
        """
        byte_array = byte_array.reshape((-1, 16))
        if endianness == "<":
            dtype = np.dtype([('second_fractions', '<u8'), ('seconds', '<i8')])
        else:
            dtype = np.dtype([('seconds', '>i8'), ('second_fractions', '>u8')])
        return TimestampArray(byte_array.view(dtype).reshape(-1))


@tds_data_type(0x08000c, np.complex64)
class ComplexSingleFloat(TdmsType):
    size = 8


@tds_data_type(0x10000d, np.complex128)
class ComplexDoubleFloat(TdmsType):
    size = 16


@tds_data_type(0xFFFFFFFF, None)
class DaqMxRawData(TdmsType):
    pass
