"""Test reading of example TDMS files"""

from collections import defaultdict
import logging
import os
import sys
from shutil import copyfile
import tempfile
import weakref
from hypothesis import (assume, given, example, settings, strategies, HealthCheck)
import numpy as np
import pytest
from nptdms import TdmsFile
from nptdms.log import log_manager
from nptdms.test.util import (
    BytesIoTestFile,
    GeneratedFile,
    basic_segment,
    channel_metadata,
    compare_arrays,
    hexlify_value,
    segment_objects_metadata,
    string_hexlify,
)
from nptdms.test import scenarios


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_read_channel_data(test_file, expected_data):
    """Test reading data"""

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file)

    for ((group, channel), expected_data) in expected_data.items():
        channel_obj = tdms_data[group][channel]
        actual_data = channel_obj.data
        assert actual_data.dtype == expected_data.dtype
        assert channel_obj.dtype == expected_data.dtype
        compare_arrays(actual_data, expected_data)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_lazily_read_channel_data(test_file, expected_data):
    """Test reading channel data lazily"""

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file[group][channel].read_data()
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data)


def test_read_raw_channel_data():
    """Test reading raw channel data"""

    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        for ((group, channel), expected_data) in expected_data.items():
            actual_data = tdms_file[group][channel].read_data(scaled=False)
            assert actual_data.dtype == expected_data.dtype
            compare_arrays(actual_data, expected_data)


def test_lazily_read_raw_channel_data():
    """Test reading raw channel data lazily"""

    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file[group][channel].read_data(scaled=False)
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data)


def test_read_raw_channel_data_slice():
    """Test reading a slice of raw channel data"""

    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        for ((group, channel), expected_data) in expected_data.items():
            actual_data = tdms_file[group][channel].read_data(offset=1, length=2, scaled=False)
            assert actual_data.dtype == expected_data.dtype
            compare_arrays(actual_data, expected_data[1:3])


def test_lazily_read_raw_channel_data_slice():
    """Test reading raw channel data lazily"""

    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file[group][channel].read_data(offset=1, length=2, scaled=False)
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data[1:3])


def test_lazily_read_channel_data_with_file_path():
    """Test reading channel data lazily after initialising with a file path
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with TdmsFile.open(temp_file.name) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file[group][channel].read_data()
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data)
    finally:
        os.remove(temp_file.name)


@given(offset=strategies.integers(0, 100), length=strategies.integers(0, 100))
@example(offset=0, length=0)
@example(offset=0, length=100)
@example(offset=0, length=5)
@example(offset=0, length=10)
def test_reading_subset_of_data(offset, length):
    channel_data = np.arange(0, 100, 1, dtype=np.int32)
    # Split data into different sized segments
    segment_data = [
        channel_data[0:10],
        channel_data[10:20],
        channel_data[20:60],
        channel_data[60:80],
        channel_data[80:90],
        channel_data[90:100],
    ]
    hex_segment_data = [
        "".join(hexlify_value('<i', x) for x in data) for data in segment_data]
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 5),
        ),
        hex_segment_data[0]
    )
    for hex_data in hex_segment_data[1:]:
        test_file.add_segment(("kTocRawData", ), "", hex_data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            channel_subset = tdms_file['group']['channel1'].read_data(offset, length)
            expected_data = channel_data[offset:offset + length]
            assert len(channel_subset) == len(expected_data)
            np.testing.assert_equal(channel_subset, expected_data)


@settings(suppress_health_check=[HealthCheck.filter_too_much])
@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
@given(offset=strategies.integers(0, 10), length=strategies.integers(0, 10))
def test_reading_subset_of_data_for_scenario(test_file, expected_data, offset, length):
    """Test reading a subset of a channel's data
    """
    assume(any(offset <= len(d) for d in expected_data.values()))
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file[group][channel].read_data(offset, length)
                compare_arrays(actual_data, expected_data[offset:offset + length])


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_stream_data_chunks(test_file, expected_data):
    """Test streaming chunks of data from a TDMS file
    """
    data_arrays = defaultdict(list)
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for chunk in tdms_file.data_chunks():
                for group in chunk.groups():
                    for channel in group.channels():
                        key = (group.name, channel.name)
                        assert channel.offset == len(data_arrays[key])
                        data_arrays[key].extend(channel[:])

    for ((group, channel), expected_data) in expected_data.items():
        actual_data = data_arrays[(group, channel)]
        compare_arrays(actual_data, expected_data)


def test_indexing_and_iterating_data_chunks():
    """Test streaming chunks of data from a TDMS file and indexing into chunks
    """
    test_file, expected_data = scenarios.single_segment_with_two_channels().values
    data_arrays = defaultdict(list)
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for chunk in tdms_file.data_chunks():
                for (group, channel) in expected_data.keys():
                    key = (group, channel)
                    channel_chunk = chunk[group][channel]
                    data_arrays[key].extend(list(channel_chunk))

    for ((group, channel), expected_data) in expected_data.items():
        actual_data = data_arrays[(group, channel)]
        compare_arrays(actual_data, expected_data)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_stream_channel_data_chunks(test_file, expected_data):
    """Test streaming chunks of data for a single channel from a TDMS file
    """
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                actual_data = []
                for chunk in tdms_file[group][channel].data_chunks():
                    assert chunk.offset == len(actual_data)
                    actual_data.extend(chunk[:])
                compare_arrays(actual_data, expected_channel_data)


def test_iterate_channel_data_in_open_mode():
    """Test iterating over channel data after opening a file without reading data
    """
    test_file, expected_data = scenarios.chunked_segment().values

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                actual_data = []
                for value in tdms_file[group][channel]:
                    actual_data.append(value)
                compare_arrays(actual_data, expected_channel_data)


def test_iterate_channel_data_in_read_mode():
    """Test iterating over channel data after reading all data
    """
    test_file, expected_data = scenarios.chunked_segment().values

    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        for ((group, channel), expected_channel_data) in expected_data.items():
            actual_data = []
            for value in tdms_file[group][channel]:
                actual_data.append(value)
            compare_arrays(actual_data, expected_channel_data)


def test_iterate_file_and_groups():
    """ Test iterating over TdmsFile and TdmsGroup uses key values
    """
    test_file, expected_data = scenarios.chunked_segment().values

    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        for group_name in tdms_file:
            group = tdms_file[group_name]
            for channel_name in group:
                channel = group[channel_name]
                expected_channel_data = expected_data[(group_name, channel_name)]
                compare_arrays(channel.data, expected_channel_data)


def test_indexing_channel_after_read_data():
    """ Test indexing into a channel after reading all data
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
    for ((group, channel), expected_channel_data) in expected_data.items():
        channel_object = tdms_file[group][channel]
        assert channel_object[0] == expected_channel_data[0]
        compare_arrays(channel_object[:], expected_channel_data)


@given(index=strategies.integers(0, 7))
def test_indexing_channel_with_integer(index):
    """ Test indexing into a channel with an integer index
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                assert channel_object[index] == expected_channel_data[index]


@given(index=strategies.integers(-8, -1))
def test_indexing_channel_with_negative_integer(index):
    """ Test indexing into a channel with a negative integer index
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                assert channel_object[index] == expected_channel_data[index]


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_indexing_channel_with_integer_and_caching(test_file, expected_data):
    """ Test indexing into a channel with an integer index, reusing the same file to test caching
    """
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                values = []
                for i in range(len(channel_object)):
                    values.append(channel_object[i])
                compare_arrays(values, expected_channel_data)


@given(index=strategies.integers(0, 1))
def test_indexing_timestamp_channel_with_integer(index):
    """ Test indexing into a timestamp data channel with an integer index
    """
    test_file, expected_data = scenarios.timestamp_data().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                assert channel_object[index] == expected_channel_data[index]


def test_indexing_scaled_channel_with_integer():
    """ Test indexing into a channel with an integer index when the channel is scaled
    """
    test_file, expected_data = scenarios.scaled_data().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                values = []
                for i in range(len(channel_object)):
                    values.append(channel_object[i])
                compare_arrays(values, expected_channel_data)


def test_indexing_channel_with_ellipsis():
    """ Test indexing into a channel with ellipsis returns all data
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                compare_arrays(channel_object[...], expected_channel_data)


@pytest.fixture(scope="module")
def opened_tdms_file():
    """ Allow re-use of an opened TDMS file
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            yield tdms_file, expected_data


@given(
    start=strategies.integers(-10, 10) | strategies.none(),
    stop=strategies.integers(-10, 10) | strategies.none(),
    step=strategies.integers(-5, 5).filter(lambda i: i != 0) | strategies.none(),
)
@settings(max_examples=1000)
def test_indexing_channel_with_slice(opened_tdms_file, start, stop, step):
    """ Test indexing into a channel with a slice
    """
    tdms_file, expected_data = opened_tdms_file
    for ((group, channel), expected_channel_data) in expected_data.items():
        channel_object = tdms_file[group][channel]
        compare_arrays(channel_object[start:stop:step], expected_channel_data[start:stop:step])


@pytest.mark.parametrize('index', [-9, 8])
def test_indexing_channel_with_invalid_integer_raises_error(index):
    """ Test indexing into a channel with an invalid integer index
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                with pytest.raises(IndexError):
                    _ = channel_object[index]


def test_indexing_channel_with_zero_step_raises_error():
    """ Test indexing into a channel with a slice with zero step size raises an error
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                with pytest.raises(ValueError) as exc_info:
                    _ = channel_object[::0]
                assert str(exc_info.value) == "Step size cannot be zero"


@pytest.mark.parametrize('index', ["test", None])
def test_indexing_channel_with_invalid_type_raises_error(index):
    """ Test indexing into a channel with an invalid index type
    """
    test_file, expected_data = scenarios.chunked_segment().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_channel_data) in expected_data.items():
                channel_object = tdms_file[group][channel]
                with pytest.raises(TypeError) as exc_info:
                    _ = channel_object[index]
                assert "Invalid index type" in str(exc_info.value)


def test_invalid_offset_in_read_data_throws():
    """ Exception is thrown when reading a subset of data with an invalid offset
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(ValueError) as exc_info:
                tdms_file[group][channel].read_data(-1, 5)
            assert "offset must be non-negative" in str(exc_info.value)


def test_invalid_length_in_read_data_throws():
    """ Exception is thrown when reading a subset of data with an invalid length
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(ValueError) as exc_info:
                tdms_file[group][channel].read_data(0, -5)
            assert "length must be non-negative" in str(exc_info.value)


def test_read_data_after_close_throws():
    """ Trying to read after opening and closing without reading data should throw
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            pass
        with pytest.raises(RuntimeError) as exc_info:
            tdms_file[group][channel].read_data()
        assert "Cannot read data after the underlying TDMS reader is closed" in str(exc_info.value)


def test_access_data_property_after_opening_throws():
    """ Accessing the data property after opening without reading data should throw
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file[group][channel].data
            assert "Channel data has not been read" in str(exc_info.value)

            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file[group][channel].raw_data
            assert "Channel data has not been read" in str(exc_info.value)

            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file[group][channel].raw_scaler_data
            assert "Channel data has not been read" in str(exc_info.value)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_read_with_index_file(test_file, expected_data):
    """ Test reading a file with an associated tdms_index file
    """
    with test_file.get_tempfile_with_index() as tdms_file_path:
        tdms_file = TdmsFile.read(tdms_file_path)

    for ((group, channel), expected_channel_data) in expected_data.items():
        channel_obj = tdms_file[group][channel]
        compare_arrays(channel_obj.data, expected_channel_data)


@pytest.mark.skipif(sys.version_info < (3, 4), reason="pathlib only available in stdlib since 3.4")
def test_read_file_passed_as_pathlib_path():
    """ Test reading a file when using a pathlib Path object
    """
    import pathlib

    test_file, expected_data = scenarios.single_segment_with_one_channel().values

    with test_file.get_tempfile_with_index() as tdms_file_path_str:
        tdms_file_path = pathlib.Path(tdms_file_path_str)
        tdms_file = TdmsFile.read(tdms_file_path)

    for ((group, channel), expected_channel_data) in expected_data.items():
        channel_obj = tdms_file[group][channel]
        compare_arrays(channel_obj.data, expected_channel_data)


def test_read_with_mismatching_index_file():
    """ Test that reading data when the index file doesn't match the data file raises an error
    """

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2),
            channel_metadata("/'group'/'channel2'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2),
            channel_metadata("/'group'/'channel2'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )

    test_file_with_index = GeneratedFile()
    test_file_with_index.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 3),
            channel_metadata("/'group'/'channel2'", 3, 3),
        ),
        "01 00 00 00" "02 00 00 00" "03 00 00 00"
        "04 00 00 00" "05 00 00 00" "06 00 00 00"
    )
    test_file_with_index.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 3),
            channel_metadata("/'group'/'channel2'", 3, 3),
        ),
        "01 00 00 00" "02 00 00 00" "03 00 00 00"
        "04 00 00 00" "05 00 00 00" "06 00 00 00"
    )

    with test_file.get_tempfile(delete=False) as tdms_file:
        with test_file_with_index.get_tempfile_with_index() as tdms_file_with_index_path:
            # Move index file from second file to match the name of the first file
            new_index_file = tdms_file.name + '_index'
            copyfile(tdms_file_with_index_path + '_index', new_index_file)
            try:
                tdms_file.file.close()
                with pytest.raises(ValueError) as exc_info:
                    _ = TdmsFile.read(tdms_file.name)
                assert 'Check that the tdms_index file matches the tdms data file' in str(exc_info.value)
            finally:
                os.remove(new_index_file)
                os.remove(tdms_file.name)


def test_get_len_of_file():
    """Test getting the length of a TdmsFile
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    assert len(tdms_data) == 1


def test_get_len_of_group():
    """Test getting the length of a TdmsGroup
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    assert len(tdms_data['Group']) == 2


def test_key_error_getting_invalid_group():
    """Test getting a group that doesn't exist raises a KeyError
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    with pytest.raises(KeyError) as exc_info:
        _ = tdms_data['non-existent group']
    assert 'non-existent group' in str(exc_info.value)


def test_key_error_getting_invalid_channel():
    """Test getting a channel that doesn't exist raises a KeyError
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    group = tdms_data['Group']
    with pytest.raises(KeyError) as exc_info:
        _ = group['non-existent channel']
    assert 'non-existent channel' in str(exc_info.value)
    assert 'Group' in str(exc_info.value)


def test_group_property_read():
    """Test reading property of a group"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    group = tdms_data["Group"]
    assert group.properties["num"] == 10


def test_time_track():
    """Add a time track to waveform data"""

    test_file = GeneratedFile()
    (toc, metadata, data) = basic_segment()
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    channel = tdms_data["Group"]["Channel2"]
    time = channel.time_track()
    assert len(time) == len(channel.data)
    epsilon = 1.0E-15
    assert abs(time[0]) < epsilon
    assert abs(time[1] - 0.1) < epsilon


def test_memmapped_read():
    """Test reading data into memmapped arrays"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load(memmap_dir=tempfile.gettempdir())

    data = tdms_data["Group"]["Channel1"].data
    assert len(data) == 2
    assert data[0] == 1
    assert data[1] == 2
    data = tdms_data["Group"]["Channel2"].data
    assert len(data) == 2
    assert data[0] == 3
    assert data[1] == 4


def test_string_data():
    """Test reading a file with string data"""

    strings = ["abcdefg", "qwertyuiop"]

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "18 00 00 00")
    metadata += string_hexlify("/'Group'/'StringChannel'")
    metadata += (
        # Length of index information
        "1C 00 00 00"
        # Raw data data type
        "20 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of bytes in data
        "19 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "07 00 00 00"  # index to after first string
        "11 00 00 00"  # index to after second string
    )
    for string in strings:
        data += string_hexlify(string)
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    channel = tdms_data["Group"]["StringChannel"]
    assert len(channel.data) == len(strings)
    assert channel.data.dtype == channel.dtype
    for expected, read in zip(strings, channel.data):
        assert expected == read


def test_string_data_in_interleaved_segment():
    """ Test reading a file with string data where a segment
        has the interleaved data flag set but only one channel.
        (See issue #239)
    """

    strings = ["abcdefg", "qwertyuiop"]

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData")
    metadata = (
        "01 00 00 00"  # Number of objects
        "18 00 00 00")  # Length of the object path
    metadata += string_hexlify("/'Group'/'StringChannel'")
    metadata += (
        "1C 00 00 00"  # Length of index information
        "20 00 00 00"  # Raw data data type
        "01 00 00 00"  # Dimension
        "02 00 00 00 00 00 00 00"  # Number of raw data values
        "19 00 00 00 00 00 00 00"  # Number of bytes in data
        "00 00 00 00")  # Number of properties (0)
    data = (
        "07 00 00 00"  # index to after first string
        "11 00 00 00"  # index to after second string
    )
    for string in strings:
        data += string_hexlify(string)
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    channel = tdms_data["Group"]["StringChannel"]
    assert len(channel.data) == len(strings)
    assert channel.data.dtype == channel.dtype
    for expected, read in zip(strings, channel.data):
        assert expected == read


def test_incomplete_segment_with_string_data():
    """ Test incomplete last segment, eg. if LabView crashed, with string data
    """
    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "18 00 00 00")
    metadata += string_hexlify("/'Group'/'StringChannel'")
    metadata += (
        # Length of index information
        "1C 00 00 00"
        # Raw data data type
        "20 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of bytes in data
        "19 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = "00 00 00 00"
    test_file.add_segment(toc, metadata, data, incomplete=True)

    tdms_data = test_file.load()

    channel = tdms_data["Group"]["StringChannel"]
    assert len(channel) == 0


def test_slash_and_space_in_name():
    """Test name like '01/02/03 something'"""

    group_1 = "01/02/03 something"
    channel_1 = "04/05/06 another thing"
    group_2 = "01/02/03 a"
    channel_2 = "04/05/06 b"

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'{0}'/'{1}'".format(group_1, channel_1), 3, 2),
            channel_metadata("/'{0}'/'{1}'".format(group_2, channel_2), 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )

    tdms_data = test_file.load()

    assert len(tdms_data.groups()) == 2
    assert len(tdms_data[group_1].channels()) == 1
    assert len(tdms_data[group_2].channels()) == 1
    data_1 = tdms_data[group_1][channel_1].data
    assert len(data_1) == 2
    data_2 = tdms_data[group_2][channel_2].data
    assert len(data_2) == 2


def test_single_quote_in_name():
    group_1 = "group''s name"
    channel_1 = "channel''s name"

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'{0}'/'{1}'".format(group_1, channel_1), 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
    )

    tdms_data = test_file.load()

    assert len(tdms_data.groups()) == 1
    assert len(tdms_data["group's name"].channels()) == 1
    data_1 = tdms_data["group's name"]["channel's name"].data
    assert len(data_1) == 2


def test_group_object_paths():
    """Test the path and name properties for a group"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    group = tdms_data["Group"]
    assert group.path == "/'Group'"
    assert group.name == "Group"


def test_channel_object_paths():
    """Test the path and name properties for a channel"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    channel = tdms_data["Group"]["Channel1"]
    assert channel.path == "/'Group'/'Channel1'"
    assert channel.name == "Channel1"
    assert channel.group_name == "Group"


def test_object_repr():
    """Test getting object representations of groups and channels
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    group = tdms_data["Group"]
    assert repr(group) == "<TdmsGroup with path /'Group'>"

    channel = group["Channel1"]
    assert repr(channel) == "<TdmsChannel with path /'Group'/'Channel1'>"


def test_data_read_from_bytes_io():
    """Test reading data"""

    test_file = BytesIoTestFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].data
    assert len(data) == 2
    assert data[0] == 1
    assert data[1] == 2
    data = tdms_data["Group"]["Channel2"].data
    assert len(data) == 2
    assert data[0] == 3
    assert data[1] == 4


def test_file_properties():
    """Test reading properties of the file (root object)"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())

    tdms_file = test_file.load()

    file_props = tdms_file.properties
    assert file_props['num'] == 15


def test_debug_logging(caplog):
    """ Test loading a file with debug logging enabled
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values

    log_manager.set_level(logging.DEBUG)
    _ = test_file.load()

    assert "Reading metadata for object /'group'/'channel1' with index header 0x00000014" in caplog.text
    assert "Object data type: Int32" in caplog.text


def test_memory_released_when_tdms_file_out_of_scope():
    """ Tests that when a TDMS file object goes out of scope,
        TDMS channels and their data are also freed.
        This ensures there are no circular references between a TDMS file
        and its channels, which would mean the GC is needed to free these objects.
    """

    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file)
        chan = tdms_data['group']['channel1']
        chan_ref = weakref.ref(chan)
        data_ref = weakref.ref(chan.data)
        raw_data_ref = weakref.ref(chan.raw_data)
    del tdms_data
    del chan

    assert raw_data_ref() is None
    assert data_ref() is None
    assert chan_ref() is None


def test_close_after_read():
    test_file, _ = scenarios.single_segment_with_one_channel().values
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        tdms_data = TdmsFile.read(temp_file.name)
        tdms_data.close()
    finally:
        os.remove(temp_file.name)


def test_multiple_close_after_open():
    test_file, _ = scenarios.single_segment_with_one_channel().values
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with TdmsFile.open(temp_file.name) as tdms_data:
            tdms_data.close()
        tdms_data.close()
    finally:
        os.remove(temp_file.name)


def test_interleaved_segment_different_length():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 3),
            channel_metadata("/'group'/'channel2'", 3, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "01 00 00 00" "02 00 00 00"
        "01 00 00 00"
    )

    with pytest.raises(ValueError) as exc_info:
        _ = test_file.load()
    assert str(exc_info.value) == "Cannot read interleaved data with different chunk sizes"


def test_multiple_string_channels_with_interleaved_flag():
    """ Test reading a segment with the interleaved data flag set but multiple string channels throws an exception
    """

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData")
    metadata = "02 00 00 00"  # Number of objects
    for channel in ["Channel1", "Channel2"]:
        channel_path = "/'Group'/'%s'" % channel
        metadata += hexlify_value('<L', len(channel_path))
        metadata += string_hexlify(channel_path)
        metadata += (
            "1C 00 00 00"  # Length of index information
            "20 00 00 00"  # Raw data data type
            "01 00 00 00"  # Dimension
            "02 00 00 00 00 00 00 00"  # Number of raw data values
            "02 00 00 00 00 00 00 00"  # Number of bytes in data
            "00 00 00 00")  # Number of properties
    data = "01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00"
    test_file.add_segment(toc, metadata, data)
    with pytest.raises(ValueError) as exc_info:
        _ = test_file.load()
    assert str(exc_info.value) == "Cannot read interleaved segment containing channels with unsized types"


def test_interleaved_segment_with_sized_and_unsized_types():
    test_file = GeneratedFile()
    string_channel_metadata = (
        "18 00 00 00" +  # Length of the object path
        string_hexlify("/'group'/'StringChannel'") +
        "1C 00 00 00"  # Length of index information
        "20 00 00 00"  # Raw data data type
        "01 00 00 00"  # Dimension
        "02 00 00 00 00 00 00 00"  # Number of raw data values
        "02 00 00 00 00 00 00 00"  # Number of bytes in data
        "00 00 00 00")  # Number of properties (0)
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2),
            string_channel_metadata,
        ),
        "01 00 00 00" "02 00 00 00"
        "01 00 00 00" "02 00 00 00"
    )

    with pytest.raises(ValueError) as exc_info:
        _ = test_file.load()
    assert str(exc_info.value) == "Cannot read interleaved segment containing channels with unsized types"


def test_no_new_obj_list_for_first_segment():
    """ Test when kTocNewObjList is not set in the first segment
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 4),
        ),
        "01 00 00 00" "02 00 00 00" "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 4),
        ),
        "05 00 00 00" "06 00 00 00" "07 00 00 00" "08 00 00 00"
    )

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file)
    channel_data = tdms_data['group']['channel1']
    compare_arrays(channel_data[:], [1, 2, 3, 4, 5, 6, 7, 8])
