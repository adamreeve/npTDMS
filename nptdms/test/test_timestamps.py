""" Test reading timestamp properties and data
"""

from datetime import datetime
import numpy as np
import struct
import pytest
from nptdms import TdmsFile
from nptdms.timestamp import TdmsTimestamp, TimestampArray
from nptdms.types import TimeStamp
from nptdms.test.util import (
    GeneratedFile,
    channel_metadata,
    hexlify_value,
    segment_objects_metadata,
)


def test_read_raw_timestamp_properties():
    """ Test reading timestamp properties as a raw TDMS timestamp
    """
    test_file = GeneratedFile()
    second_fractions = 1234567890 * 10 ** 10
    properties = {
        "wf_start_time": (0x44, hexlify_value("<Q", second_fractions) + hexlify_value("<q", 3524551547))
    }
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2, properties),
        ),
        "01 00 00 00" "02 00 00 00"
    )

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file, raw_timestamps=True)
        start_time = tdms_data['group']['channel1'].properties['wf_start_time']
        assert start_time.seconds == 3524551547
        assert start_time.second_fractions == second_fractions


def test_timestamp_as_datetime64():
    """ Test converting a timestamp to a numpy datetime64
    """
    second_fractions = 1234567890 * 10 ** 10
    seconds = 3524551547
    timestamp = TdmsTimestamp(seconds, second_fractions)

    assert timestamp.as_datetime64() == np.datetime64('2015-09-08T10:05:47.669260', 'us')
    assert timestamp.as_datetime64().dtype == np.dtype('datetime64[us]')
    assert timestamp.as_datetime64('ns') == np.datetime64('2015-09-08T10:05:47.669260594', 'ns')
    assert timestamp.as_datetime64('ns').dtype == np.dtype('datetime64[ns]')


def test_timestamp_as_datetime():
    """ Test converting a timestamp to a datetime.datetime
    """
    second_fractions = 1234567890 * 10 ** 10
    seconds = 3524551547
    timestamp = TdmsTimestamp(seconds, second_fractions)

    assert timestamp.as_datetime() == datetime(2015, 9, 8, 10, 5, 47, 669261)


def test_read_raw_timestamp_data():
    """ Test reading timestamp data as a raw TDMS timestamps
    """
    test_file = GeneratedFile()
    seconds = 3672033330
    second_fractions = 1234567890 * 10 ** 10
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 0x44, 4),
        ),
        hexlify_value("<Q", 0) + hexlify_value("<q", seconds) +
        hexlify_value("<Q", second_fractions) + hexlify_value("<q", seconds) +
        hexlify_value("<Q", 0) + hexlify_value("<q", seconds + 1) +
        hexlify_value("<Q", second_fractions) + hexlify_value("<q", seconds + 1)
    )

    expected_seconds = np.array([seconds, seconds, seconds + 1, seconds + 1], np.dtype('int64'))
    expected_second_fractions = np.array([0, second_fractions, 0, second_fractions], np.dtype('uint64'))

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file, raw_timestamps=True)
        data = tdms_data['group']['channel1'][:]
        assert isinstance(data, TimestampArray)
        np.testing.assert_equal(data.seconds, expected_seconds)
        np.testing.assert_equal(data.second_fractions, expected_second_fractions)


def test_read_big_endian_timestamp_data():
    seconds = 3672033330
    second_fractions = 1234567890 * 10 ** 10
    data = (
        struct.pack(">q", seconds) + struct.pack(">Q", 0) +
        struct.pack(">q", seconds) + struct.pack(">Q", second_fractions) +
        struct.pack(">q", seconds + 1) + struct.pack(">Q", 0) +
        struct.pack(">q", seconds + 1) + struct.pack(">Q", second_fractions))
    expected_seconds = np.array([seconds, seconds, seconds + 1, seconds + 1], np.dtype('int64'))
    expected_second_fractions = np.array([0, second_fractions, 0, second_fractions], np.dtype('uint64'))

    timestamp_array = TimeStamp.from_bytes(np.frombuffer(data, dtype=np.dtype('uint8')), '>')

    np.testing.assert_equal(timestamp_array.seconds, expected_seconds)
    np.testing.assert_equal(timestamp_array.second_fractions, expected_second_fractions)


def test_timestamp_repr():
    timestamp = TdmsTimestamp(3672033330, 12345678900000000000)
    assert repr(timestamp) == 'TdmsTimestamp(3672033330, 12345678900000000000)'


def test_timestamp_str():
    timestamp = TdmsTimestamp(3672033330, 12345678900000000000)
    assert str(timestamp) == '2020-05-11T09:15:30.669261'


def test_timestamp_array_slicing():
    timestamp_array = _get_test_timestamp_array()
    array_slice = timestamp_array[0:2]
    assert isinstance(array_slice, TimestampArray)


def test_timestamp_array_get_single_item():
    timestamp_array = _get_test_timestamp_array()
    array_item = timestamp_array[3]
    assert isinstance(array_item, TdmsTimestamp)
    assert array_item.seconds == 3672033331
    assert array_item.second_fractions == 1234567890 * 10 ** 10


def test_timestamp_array_field_access():
    timestamp_array = _get_test_timestamp_array()
    seconds = timestamp_array.seconds
    second_fractions = timestamp_array.second_fractions
    assert isinstance(seconds, np.ndarray)
    assert not isinstance(seconds, TimestampArray)
    assert seconds.dtype == np.dtype('<i8')
    assert isinstance(second_fractions, np.ndarray)
    assert not isinstance(second_fractions, TimestampArray)
    assert second_fractions.dtype == np.dtype('<u8')


def test_timestamp_array_to_datetime64():
    timestamp_array = _get_test_timestamp_array()
    expected_timestamps = np.array([
        np.datetime64('2020-05-11 09:15:30'),
        np.datetime64('2020-05-11 09:15:30.669260'),
        np.datetime64('2020-05-11 09:15:31'),
        np.datetime64('2020-05-11 09:15:31.669260'),
    ])

    us_array = timestamp_array.as_datetime64()

    np.testing.assert_equal(us_array, expected_timestamps)


def test_timestamp_array_to_datetime64_with_ns_precision():
    timestamp_array = _get_test_timestamp_array()
    expected_timestamps = np.array([
        np.datetime64('2020-05-11 09:15:30'),
        np.datetime64('2020-05-11 09:15:30.669260594'),
        np.datetime64('2020-05-11 09:15:31'),
        np.datetime64('2020-05-11 09:15:31.669260594'),
    ])

    ns_array = timestamp_array.as_datetime64('ns')

    np.testing.assert_equal(ns_array, expected_timestamps)


@pytest.mark.parametrize(
    "input_array",
    [
        np.array([1, 2, 3, 4]),
        np.array([(1, 2), (3, 4)], dtype=[('a', '<i8'), ('b', '<u8')]),
        np.array([(1, 2), (3, 4)], dtype=[('seconds', '<i8'), ('b', '<u8')]),
        np.array([(1, 2), (3, 4)], dtype=[('a', '<i8'), ('second_fractions', '<u8')]),
    ]
)
def test_error_raised_with_creating_timestamp_array_with_invalid_input_type(input_array):
    with pytest.raises(ValueError) as exc_info:
        _ = TimestampArray(input_array)
    assert str(exc_info.value) == "Input array must have a dtype with 'seconds' and 'second_fractions' fields"


def test_error_raised_converting_timestamp_with_invalid_resolution():
    timestamp_array = _get_test_timestamp_array()
    timestamp = timestamp_array[0]
    with pytest.raises(ValueError) as exc_info:
        _ = timestamp.as_datetime64('invalid_res')
    assert str(exc_info.value) == "Unsupported resolution for converting to numpy datetime64: 'invalid_res'"


def test_error_raised_converting_timestamp_array_with_invalid_resolution():
    timestamp_array = _get_test_timestamp_array()
    with pytest.raises(ValueError) as exc_info:
        _ = timestamp_array.as_datetime64('invalid_res')
    assert str(exc_info.value) == "Unsupported resolution for converting to numpy datetime64: 'invalid_res'"


def _get_test_timestamp_array():
    dtype = np.dtype([('seconds', '<i8'), ('second_fractions', '<u8')])
    seconds = 3672033330
    second_fractions = 1234567890 * 10 ** 10
    array = np.array([
        (seconds, 0),
        (seconds, second_fractions),
        (seconds + 1, 0),
        (seconds + 1, second_fractions),
    ], dtype=dtype)
    return TimestampArray(array)
