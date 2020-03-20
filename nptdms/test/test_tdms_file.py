"""Test reading of example TDMS files"""

import os
import tempfile
from hypothesis import (assume, given, example, strategies)
import numpy as np
import pytest
from nptdms import TdmsFile
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
        actual_data = tdms_data.object(group, channel).data
        assert actual_data.dtype == expected_data.dtype
        compare_arrays(actual_data, expected_data)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_lazily_read_channel_data(test_file, expected_data):
    """Test reading channel data lazily"""

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file.object(group, channel).read_data()
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data)


def test_lazily_read_channel_data_with_file_path():
    """Test reading channel data lazily after initialising with a file path
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    temp_file = test_file.get_tempfile(delete=False)
    try:
        temp_file.file.close()
        with TdmsFile.open(temp_file.name) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file.object(group, channel).read_data()
                assert actual_data.dtype == expected_data.dtype
                compare_arrays(actual_data, expected_data)
    finally:
        os.remove(temp_file.name)


def test_lazily_read_channel_data_with_channel_data_method():
    """Test reading channel data lazily using the channel_data method of TdmsFile
    """
    test_file, expected_data = scenarios.single_segment_with_two_channels().values
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file.channel_data(group, channel)
                assert actual_data.dtype == expected_data.dtype
                np.testing.assert_almost_equal(actual_data, expected_data)


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
            channel_subset = tdms_file.object('group', 'channel1').read_data(offset, length)
            expected_data = channel_data[offset:offset + length]
            assert len(channel_subset) == len(expected_data)
            np.testing.assert_equal(channel_subset, expected_data)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
@given(offset=strategies.integers(0, 10), length=strategies.integers(0, 10))
def test_reading_subset_of_data_for_scenario(test_file, expected_data, offset, length):
    """Test reading a subset of a channel's data
    """
    assume(any(offset <= len(d) for d in expected_data.values()))
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file.object(group, channel).read_data(offset, length)
                compare_arrays(actual_data, expected_data[offset:offset + length])


def test_invalid_offset_throws():
    """ Exception is thrown when reading a subset of data with an invalid offset
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(ValueError) as exc_info:
                tdms_file.object(group, channel).read_data(-1, 5)
            assert "offset must be non-negative" in str(exc_info.value)


def test_invalid_length_throws():
    """ Exception is thrown when reading a subset of data with an invalid length
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(ValueError) as exc_info:
                tdms_file.object(group, channel).read_data(0, -5)
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
            tdms_file.object(group, channel).read_data()
        assert "Cannot read channel data after the underlying TDMS reader is closed" in str(exc_info.value)


def test_read_data_after_open_in_read_mode_throws():
    """ Trying to read channel data after reading all data initially should throw
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        with pytest.raises(RuntimeError) as exc_info:
            tdms_file.object(group, channel).read_data()
        assert "Cannot read channel data after the underlying TDMS reader is closed" in str(exc_info.value)


def test_access_data_property_after_opening_throws():
    """ Accessing the data property after opening without reading data should throw
    """
    test_file, expected_data = scenarios.single_segment_with_one_channel().values
    group, channel = list(expected_data.keys())[0]
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file.object(group, channel).data
            assert "Channel data has not been read" in str(exc_info.value)

            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file.object(group, channel).raw_data
            assert "Channel data has not been read" in str(exc_info.value)

            with pytest.raises(RuntimeError) as exc_info:
                _ = tdms_file.object(group, channel).raw_scaler_data
            assert "Channel data has not been read" in str(exc_info.value)


def test_get_objects():
    """Test reading data"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_file = test_file.load()

    objects = tdms_file.objects
    assert len(objects) == 4
    assert "/" in objects.keys()
    assert "/'Group'" in objects.keys()
    assert "/'Group'/'Channel1'" in objects.keys()
    assert "/'Group'/'Channel2'" in objects.keys()


def test_property_read():
    """Test reading an object property"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object("Group")
    assert obj.property("num") == 10


def test_time_track():
    """Add a time track to waveform data"""

    test_file = GeneratedFile()
    (toc, metadata, data) = basic_segment()
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    obj = tdms_data.object("Group", "Channel2")
    time = obj.time_track()
    assert len(time) == len(obj.data)
    epsilon = 1.0E-15
    assert abs(time[0]) < epsilon
    assert abs(time[1] - 0.1) < epsilon


def test_memmapped_read():
    """Test reading data into memmapped arrays"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load(memmap_dir=tempfile.gettempdir())

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 2
    assert data[0] == 1
    assert data[1] == 2
    data = tdms_data.channel_data("Group", "Channel2")
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

    data = tdms_data.channel_data("Group", "StringChannel")
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read


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
    assert len(tdms_data.group_channels(group_1)) == 1
    assert len(tdms_data.group_channels(group_2)) == 1
    data_1 = tdms_data.channel_data(group_1, channel_1)
    assert len(data_1) == 2
    data_2 = tdms_data.channel_data(group_2, channel_2)
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
    assert len(tdms_data.group_channels("group's name")) == 1
    data_1 = tdms_data.channel_data("group's name", "channel's name")
    assert len(data_1) == 2


def test_root_object_paths():
    """Test the group and channel properties for the root object"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object()
    assert obj.group is None
    assert obj.channel is None


def test_group_object_paths():
    """Test the group and channel properties for a group"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object("Group")
    assert obj.group == "Group"
    assert obj.channel is None


def test_channel_object_paths():
    """Test the group and channel properties for a group"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object("Group", "Channel1")
    assert obj.group == "Group"
    assert obj.channel == "Channel1"


def test_data_read_from_bytes_io():
    """Test reading data"""

    test_file = BytesIoTestFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 2
    assert data[0] == 1
    assert data[1] == 2
    data = tdms_data.channel_data("Group", "Channel2")
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
