""" Contains different test cases for tests for reading TDMS files
"""
import numpy as np
import pytest

from nptdms.test.util import (
    GeneratedFile,
    hexlify_value,
    segment_objects_metadata,
    string_hexlify)


_scenarios = []


def scenario(func):
    def as_param():
        result = func()
        return pytest.param(*result, id=func.__name__)
    _scenarios.append(as_param)
    return as_param


def get_scenarios():
    return [f() for f in _scenarios]


@scenario
def single_segment_with_one_channel():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            _channel_metadata("/'group'/'channel1'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def single_segment_with_two_channels():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            _channel_metadata("/'group'/'channel1'", 3, 2),
            _channel_metadata("/'group'/'channel2'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def single_segment_with_two_channels_interleaved():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            _channel_metadata("/'group'/'channel1'", 3, 2),
            _channel_metadata("/'group'/'channel2'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 3], dtype=np.int32),
        ('group', 'channel2'): np.array([2, 4], dtype=np.int32),
    }
    return test_file, expected_data


def _channel_metadata(channel_name, data_type, num_values):
    return (
        # Length of the object path
        hexlify_value('<I', len(channel_name)) +
        # Object path
        string_hexlify(channel_name) +
        # Length of index information
        "14 00 00 00" +
        # Raw data data type
        hexlify_value('<I', data_type) +
        # Dimension
        "01 00 00 00" +
        # Number of raw data values
        hexlify_value('<Q', num_values) +
        # Number of properties (0)
        "00 00 00 00"
    )
