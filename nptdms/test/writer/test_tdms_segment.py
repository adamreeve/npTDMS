"""Test TdmsSegment"""

import pytest
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict

from nptdms.writer import TdmsSegment, read_properties_dict
from nptdms.types import *


class StubObject(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def test_write_leadin_with_one_channel():
    data_type = StubObject(size=4)

    channel = StubObject(
        path="",
        has_data=True,
        data=[0] * 10,
        data_type=data_type)

    toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
    metadata_size = 12

    segment = TdmsSegment([channel])
    leadin = segment.leadin(toc, metadata_size)

    expected_values = [
        Bytes(b'TDSm'),
        Int32(14),  # TOC bitmask
        Int32(4712),  # TDMS version
        Uint64(52),  # Next segment offset
        Uint64(12),  # Raw data offset
        ]

    _assert_sequence_equal(leadin, expected_values)


def test_write_leadin_with_object_without_data():
    channel = StubObject(
        path="",
        has_data=False)

    toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
    metadata_size = 12

    segment = TdmsSegment([channel])
    leadin = segment.leadin(toc, metadata_size)

    expected_values = [
        Bytes(b'TDSm'),
        Int32(14),  # TOC bitmask
        Int32(4712),  # TDMS version
        Uint64(12),  # Next segment offset
        Uint64(12),  # Raw data offset
        ]

    _assert_sequence_equal(leadin, expected_values)


def test_write_metadata_with_properties():
    data_type = StubObject(enum_value=3)

    # Use an ordered dict for properties so that
    # the order of properties in metadata is guaranteed
    properties = OrderedDict()
    properties["prop1"] = String("foo")
    properties["prop2"] = Int32(42)

    channel = StubObject(
        path="channel_path",
        has_data=True,
        data=[1] * 10,
        data_type=data_type,
        properties=properties)

    segment = TdmsSegment([channel])
    metadata = segment.metadata()

    expected_values = [
        Uint32(1),  # Number of objects
        String("channel_path"),
        Uint32(20),  # Length of raw data index in bytes
        Int32(3),  # Data type
        Uint32(1),  # Array dimension
        Uint64(10),  # Number of values
        Uint32(2),  # Number of properties
        String("prop1"),  # Property name
        Int32(0x20),
        String("foo"),
        String("prop2"),
        Int32(3),
        Int32(42),
        ]

    _assert_sequence_equal(metadata, expected_values)


def test_write_metadata_with_no_data():
    obj = StubObject(
        path="object_path",
        has_data=False,
        properties={})

    segment = TdmsSegment([obj])
    metadata = segment.metadata()

    expected_values = [
        Uint32(1),  # Number of objects
        String("object_path"),
        Bytes(b'\xFF\xFF\xFF\xFF'),  # Raw data index
        Uint32(0),  # Number of properties
        ]

    _assert_sequence_equal(metadata, expected_values)


def test_properties_are_converted_to_tdms_types():
    test_time = datetime.utcnow()

    properties = {
        "prop1": Int32(1),
        "prop2": 2,
        "prop3": "foo",
        "prop4": True,
        "prop5": 3.142,
        "prop6": test_time,
    }

    tdms_properties = read_properties_dict(properties)

    assert len(tdms_properties) == len(properties)
    assert tdms_properties["prop1"] == Int32(1)
    assert tdms_properties["prop2"] == Int32(2)
    assert tdms_properties["prop3"] == String("foo")
    assert tdms_properties["prop4"] == Boolean(True)
    assert tdms_properties["prop5"] == DoubleFloat(3.142)
    assert tdms_properties["prop6"] == TimeStamp(test_time)


def test_datetime_converted_when_it_only_has_date_part():
    test_time = np.datetime64('2017-11-19')

    properties = {
        "time_prop": test_time,
    }

    tdms_properties = read_properties_dict(properties)

    assert tdms_properties["time_prop"] == TimeStamp(test_time)


def test_writing_long_integer_properties():
    properties = {
        "prop1": 2147483647,
        "prop2": 2147483648,
    }

    tdms_properties = read_properties_dict(properties)

    assert len(tdms_properties) == len(properties)
    assert tdms_properties["prop1"] == Int32(2147483647)
    assert tdms_properties["prop2"] == Int64(2147483648)


def test_writing_properties_with_numpy_typed_values():
    properties = {
        "int32prop": np.int32(32),
        "int64prop": np.int64(64),
        "float32prop": np.float32(32.0),
        "float64prop": np.float64(64.0),
    }

    tdms_properties = read_properties_dict(properties)

    assert len(tdms_properties) == len(properties)
    assert tdms_properties["int32prop"] == Int32(32)
    assert tdms_properties["int64prop"] == Int64(64)
    assert tdms_properties["float32prop"] == SingleFloat(32.0)
    assert tdms_properties["float64prop"] == DoubleFloat(64.0)


def test_error_raised_when_cannot_convert_property_value():
    properties = {
        "prop1": None
    }

    with pytest.raises(TypeError):
        read_properties_dict(properties)


def _assert_sequence_equal(values, expected_values):
    position = 1
    expected_values = iter(expected_values)
    for val in values:
        try:
            expected = next(expected_values)
        except StopIteration:
            raise ValueError(
                "Expected end of sequence at position %d but found: %r" %
                (position, val))
        assert val == expected, "Expected %r to equal %r at position %d" % (val, expected, position)
        position += 1
    try:
        expected = next(expected_values)
        raise ValueError(
            "Expected %r at position %d but found end of sequence" %
            (expected, position))
    except StopIteration:
        pass
