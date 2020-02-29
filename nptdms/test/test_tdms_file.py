"""Test reading of example TDMS files"""

import numpy as np
from nptdms.test.util import *


def test_data_read():
    """Test reading data"""

    test_file = GeneratedFile()
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


def test_no_metadata_segment():
    """Add a segment with two channels, then a second
    segment with the same metadata as before,
    so there is only the lead in and binary data"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    data = (
        "05 00 00 00"
        "06 00 00 00"
        "07 00 00 00"
        "08 00 00 00"
    )
    toc = ("kTocRawData")
    test_file.add_segment('', data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 4
    assert all(data == [3, 4, 7, 8])


def test_no_metadata_object():
    """Re-use an object without setting any new metadata and
    re-using the data structure"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    data = (
        "05 00 00 00"
        "06 00 00 00"
        "07 00 00 00"
        "08 00 00 00"
    )
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    # Use same object list, but set raw data index to 0
    metadata = (
        # Number of objects
        "03 00 00 00"
        # Length of the first object path
        "08 00 00 00"
        # Object path (/'Group')
        "2F 27 47 72"
        "6F 75 70 27"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00"
        # Length of the second object path
        "13 00 00 00"
        # Second object path (/'Group'/'Channel1')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 31 27"
        # Raw data index meaning repeat previous data structure
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00"
        # Length of the third object path
        "13 00 00 00"
        # Third object path (/'Group'/'Channel2')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 32 27"
        # Raw data index meaning repeat previous data structure
        "00 00 00 00"
        # Number of properties
        "00 00 00 00")
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 4
    assert all(data == [3, 4, 7, 8])


def test_new_channel():
    """Add a new voltage channel, with the other two channels
    remaining unchanged, so only the new channel is in metadata section"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    toc = ("kTocMetaData", "kTocRawData")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the third object path
        "12 00 00 00")
    metadata += string_hexlify("/'Group'/'Voltage'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "05 00 00 00"
        "06 00 00 00"
        "07 00 00 00"
        "08 00 00 00"
        "09 00 00 00"
        "0A 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 4
    assert all(data == [3, 4, 7, 8])
    data = tdms_data.channel_data("Group", "Voltage")
    assert len(data) == 2
    assert all(data == [9, 10])


def test_larger_channel():
    """In the second segment, increase the channel size
    of one channel"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    toc = ("kTocMetaData", "kTocRawData")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel2'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "04 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "05 00 00 00"
        "06 00 00 00"
        "07 00 00 00"
        "08 00 00 00"
        "09 00 00 00"
        "0A 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()
    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 6
    assert all(data == [3, 4, 7, 8, 9, 10])


def test_remove_channel():
    """In the second segment, remove a channel.
    We need to write a new object list in this case"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    # Keep toc as it was before, with new object list set
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel1'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "05 00 00 00"
        "06 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()
    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 2
    assert all(data == [3, 4])


def test_no_lead_in():
    """Add segment and then a repeated segment without
    any lead in or metadata, so data is read in chunks"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    data = data + (
        "05 00 00 00"
        "06 00 00 00"
        "07 00 00 00"
        "08 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 4
    assert all(data == [1, 2, 5, 6])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 4
    assert all(data == [3, 4, 7, 8])


def test_interleaved():
    """Test reading interleaved data"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    toc = toc + ("kTocInterleavedData", )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 2
    assert data[0] == 1
    assert data[1] == 3
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 2
    assert data[0] == 2
    assert data[1] == 4


def test_less_data_than_expected():
    """Add segment and then a repeated segment without
    any lead in or metadata, so data is read in chunks,
    but the extra chunk does not have as much data as expected."""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    data = data + (
        "05 00 00 00"
        "06 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 3
    assert all(data == [1, 2, 5])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 3
    assert all(data == [3, 4, 6])


def test_less_data_than_expected_interleaved():
    """Add segment and then a repeated segment without
    any lead in or metadata, so data is read in chunks,
    but the extra chunk does not have as much data as expected.
    This also uses interleaved data"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    toc = toc + ("kTocInterleavedData", )
    data = data + (
        "05 00 00 00"
        "06 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 3
    assert all(data == [1, 3, 5])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 3
    assert all(data == [2, 4, 6])


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
        # Number of raw datata values
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
        # Number of raw datata values
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
    test_file.add_segment(metadata, data, toc)
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
    test_file.add_segment(metadata, data, toc)
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
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    obj = tdms_data.object("Group", "Channel2")
    time = obj.time_track()
    assert len(time) == len(obj.data)
    epsilon = 1.0E-15
    assert abs(time[0]) < epsilon
    assert abs(time[1] - 0.1) < epsilon


def test_no_data_section():
    """kTocRawData is set but data length is zero

    Keep first segment the same but add a second
    segment with no data."""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    toc = ("kTocMetaData", "kTocRawData")
    metadata = (
        # Number of objects
        "02 00 00 00"
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel1'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "00 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    metadata += (
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel2'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "00 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = ""
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()
    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 2
    assert all(data == [1, 2])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 2
    assert all(data == [3, 4])


def test_repeated_object_without_data():
    """Repeated objects with no data in new segment

    A new object is also added with new data in order
    to trigger a bug with the chunk size calculation.
    """

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "03 00 00 00"
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel1'")
    metadata += (
        # Raw data index
        "FF FF FF FF"
        # Number of properties (0)
        "00 00 00 00")
    metadata += (
        # Length of the object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel2'")
    metadata += (
        # Raw data index
        "FF FF FF FF"
        # Number of properties (0)
        "00 00 00 00"
        # Length of the third object path
        "13 00 00 00")
    metadata += string_hexlify("/'Group'/'Channel3'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "01 00 00 00"
        "02 00 00 00"
    )
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()
    data = tdms_data.channel_data("Group", "Channel1")
    assert len(data) == 2
    assert all(data == [1, 2])
    data = tdms_data.channel_data("Group", "Channel2")
    assert len(data) == 2
    assert all(data == [3, 4])
    data = tdms_data.channel_data("Group", "Channel3")
    assert len(data) == 2
    assert all(data == [1, 2])


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


def test_incomplete_data():
    """Test incomplete last segment, eg. if LabView crashed"""

    test_file = GeneratedFile()
    (metadata, data, toc) = basic_segment()
    test_file.add_segment(metadata, data, toc)
    # Add second, incomplete segment
    test_file.add_segment(metadata, data, toc, incomplete=True)
    tdms_data = test_file.load()

    # We should be able to read the incomplete segment as well as
    # previous segments
    data = tdms_data.channel_data("Group", "Channel1")
    np.testing.assert_almost_equal(data, [1, 2, 1, 2])
    data = tdms_data.channel_data("Group", "Channel2")
    np.testing.assert_almost_equal(data, [3, 4, 3, 4])


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
        # Number of raw datata values
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
    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "StringChannel")
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read


def test_complex_data():
    """Test reading a file with complex numbers data"""

    complex_single_arr = np.array([1+2j, 3+4j], dtype=np.complex64)
    complex_double_arr = np.array([5+6j, 7+8j], dtype=np.complex128)
    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "02 00 00 00"
        # Length of the object path
        "1F 00 00 00")
    metadata += string_hexlify("/'Group'/'ComplexSingleChannel'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "0C 00 08 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    metadata += ("1F 00 00 00")
    metadata += string_hexlify("/'Group'/'ComplexDoubleChannel'")
    metadata += (
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "0D 00 10 00"
        # Dimension
        "01 00 00 00"
        # Number of raw datata values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = ""
    for num in complex_single_arr:
        data += hexlify_value("<f", num.real)
        data += hexlify_value("<f", num.imag)
    for num in complex_double_arr:
        data += hexlify_value("<d", num.real)
        data += hexlify_value("<d", num.imag)

    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    data = tdms_data.channel_data("Group", "ComplexSingleChannel")
    assert len(data) == 2
    assert all(data == complex_single_arr)

    data = tdms_data.channel_data("Group", "ComplexDoubleChannel")
    assert len(data) == 2
    assert all(data == complex_double_arr)


def test_slash_and_space_in_name():
    """Test name like '01/02/03 something'"""

    group_1_name = "01/02/03 something"
    channel_1_name = "04/05/06 another thing"
    group_2_name = "01/02/03 a"
    channel_2_name = "04/05/06 b"

    test_file = GeneratedFile()

    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")

    # Number of objects
    metadata = "04 00 00 00"

    for group in [group_1_name, group_2_name]:
        path = "/'{0}'".format(group)
        metadata += hexlify_value('<l', len(path))
        metadata += string_hexlify(path)
        metadata += (
            # Raw data index
            "FF FF FF FF"
            # Number of properties (0)
            "00 00 00 00"
        )
    for (group, channel) in [
            (group_1_name, channel_1_name),
            (group_2_name, channel_2_name)]:
        path = "/'{0}'/'{1}'".format(group, channel)
        metadata += hexlify_value('<l', len(path))
        metadata += string_hexlify(path)
        metadata += (
            # Length of index information
            "14 00 00 00"
            # Raw data data type
            "03 00 00 00"
            # Dimension
            "01 00 00 00"
            # Number of raw datata values
            "02 00 00 00"
            "00 00 00 00"
            # Number of properties (0)
            "00 00 00 00"
        )

    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "03 00 00 00"
        "04 00 00 00"
    )

    test_file.add_segment(metadata, data, toc)
    tdms_data = test_file.load()

    assert len(tdms_data.groups()) == 2
    assert len(tdms_data.group_channels(group_1_name)) == 1
    assert len(tdms_data.group_channels(group_2_name)) == 1
    data_1 = tdms_data.channel_data(group_1_name, channel_1_name)
    assert len(data_1) == 2
    data_2 = tdms_data.channel_data(group_2_name, channel_2_name)
    assert len(data_2) == 2


def test_root_object_paths():
    """Test the group and channel properties for the root object"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object()
    assert obj.group == None
    assert obj.channel == None


def test_group_object_paths():
    """Test the group and channel properties for a group"""
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    obj = tdms_data.object("Group")
    assert obj.group == "Group"
    assert obj.channel == None


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
