"""Test reading of example TDMS files"""

import os
import numpy as np
import pytest
from nptdms import TdmsFile
from nptdms.test.util import *
from nptdms.test import scenarios


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_read_channel_data(test_file, expected_data):
    """Test reading data"""

    with test_file.get_tempfile() as temp_file:
        tdms_data = TdmsFile.read(temp_file.file)

    for ((group, channel), expected_data) in expected_data.items():
        actual_data = tdms_data.object(group, channel).data
        assert actual_data.dtype == expected_data.dtype
        np.testing.assert_almost_equal(actual_data, expected_data)


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_lazily_read_channel_data(test_file, expected_data):
    """Test reading channel data lazily"""

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for ((group, channel), expected_data) in expected_data.items():
                actual_data = tdms_file.object(group, channel).read_data()
                assert actual_data.dtype == expected_data.dtype
                np.testing.assert_almost_equal(actual_data, expected_data)


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
                np.testing.assert_almost_equal(actual_data, expected_data)
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

    object = tdms_data.object("Group")
    assert object.property("num") == 10


def test_timestamp_data():
    """Test reading contiguous and interleaved timestamp data,
    which isn't read by numpy"""

    times = [
        np.datetime64('2012-08-23T00:00:00.123'),
        np.datetime64('2012-08-23T01:02:03.456'),
        np.datetime64('2012-08-23T12:00:00.0'),
        np.datetime64('2012-08-23T12:02:03.9999'),
    ]
    epoch = np.datetime64('1904-01-01T00:00:00')

    def total_seconds(td):
        return int(td / np.timedelta64(1, 's'))

    def microseconds(dt):
        diff = dt - epoch
        seconds = total_seconds(diff)
        remainder = diff - np.timedelta64(seconds, 's')
        return int(remainder / np.timedelta64(1, 'us'))

    seconds = [total_seconds(t - epoch) for t in times]
    fractions = [
        int(float(microseconds(t)) * 2 ** 58 / 5 ** 6)
        for t in times]

    metadata = (
        # Number of objects
        "02 00 00 00"
        # Length of the object path
        "17 00 00 00")
    metadata += string_hexlify("/'Group'/'TimeChannel1'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "44 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    metadata += (
        "17 00 00 00")
    metadata += string_hexlify("/'Group'/'TimeChannel2'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "44 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = ""
    for f, s in zip(fractions, seconds):
        data += hexlify_value("<Q", f)
        data += hexlify_value("<q", s)

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()
    channel_data = tdms_data.channel_data("Group", "TimeChannel1")
    assert len(channel_data) == 2
    assert channel_data[0] == times[0]
    assert channel_data[1] == times[1]
    # Read fraction of second
    channel_data = tdms_data.channel_data("Group", "TimeChannel2")
    assert len(channel_data) == 2
    assert channel_data[0] == times[2]
    assert channel_data[1] == times[3]

    # Now test it interleaved
    toc = toc + ("kTocInterleavedData", )
    test_file = GeneratedFile()
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()
    channel_data = tdms_data.channel_data("Group", "TimeChannel1")
    assert len(channel_data) == 2
    assert channel_data[0] == times[0]
    assert channel_data[1] == times[2]
    channel_data = tdms_data.channel_data("Group", "TimeChannel2")
    assert len(channel_data) == 2
    assert channel_data[0] == times[1]
    assert channel_data[1] == times[3]


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
