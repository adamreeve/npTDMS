from collections import namedtuple
import numpy as np


try:
    long
except NameError:
    # Python 3
    long = int


# Class for describing data types, with TDMS enum value, data type name,
# identifier used by struct module, the size in bytes to read and the
# numpy data type where applicable/implemented
DataType = namedtuple(
    "DataType", ('enum_value', 'name', 'struct', 'size', 'nptype'))


tds_data_types = [
    DataType(0, 'tdsTypeVoid', None, 0, None),
    DataType(1, 'tdsTypeI8', 'b', 1, np.int8),
    DataType(2, 'tdsTypeI16', 'h', 2, np.int16),
    DataType(3, 'tdsTypeI32', 'l', 4, np.int32),
    DataType(4, 'tdsTypeI64', 'q', 8, np.int64),
    DataType(5, 'tdsTypeU8', 'B', 1, np.uint8),
    DataType(6, 'tdsTypeU16', 'H', 2, np.uint16),
    DataType(7, 'tdsTypeU32', 'L', 4, np.uint32),
    DataType(8, 'tdsTypeU64', 'Q', 8, np.uint64),
    DataType(9, 'tdsTypeSingleFloat', 'f', 4, np.single),
    DataType(10, 'tdsTypeDoubleFloat', 'd', 8, np.double),
    DataType(11, 'tdsTypeExtendedFloat', None, None, None),
    DataType(12, 'tdsTypeDoubleFloatWithUnit', None, 8, None),
    DataType(13, 'tdsTypeExtendedFloatWithUnit', None, None, None),
    DataType(0x19, 'tdsTypeSingleFloatWithUnit', None, 4, None),
    DataType(0x20, 'tdsTypeString', None, None, None),
    DataType(0x21, 'tdsTypeBoolean', 'b', 1, np.bool8),
    DataType(0x44, 'tdsTypeTimeStamp', 'Qq', 16, None),
    DataType(0xFFFFFFFF, 'tdsTypeDAQmxRawData', None, 2, np.int16),
    ]


toc_properties = {
    'kTocMetaData': (long(1) << 1),
    'kTocRawData': (long(1) << 3),
    'kTocDAQmxRawData': (long(1) << 7),
    'kTocInterleavedData': (long(1) << 5),
    'kTocBigEndian': (long(1) << 6),
    'kTocNewObjList': (long(1) << 2)
}
