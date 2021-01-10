""" Test exporting TDMS data to HDF
"""
import pytest
import numpy as np
try:
    import h5py
except ImportError:
    pytest.skip("Skipping HDF tests as h5py is not installed", allow_module_level=True)

from nptdms import TdmsFile
from nptdms.test.util import (
    GeneratedFile,
    basic_segment,
    channel_metadata,
    compare_arrays,
    hexlify_value,
    segment_objects_metadata,
    string_hexlify,
)
from nptdms.test import scenarios


def test_hdf_channel_data(tmp_path):
    """ Test basic conversion of channel data to HDF
    """
    test_file, expected_data = scenarios.single_segment_with_two_channels().values

    tdms_data = test_file.load()
    h5_path = tmp_path / 'h5_data_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    for ((group, channel), expected_data) in expected_data.items():
        h5_channel = h5[group][channel]
        assert h5_channel.dtype.kind == 'i'
        np.testing.assert_almost_equal(h5_channel[...], expected_data)
    h5.close()


def test_streaming_to_hdf(tmp_path):
    """ Test conversion of channel data to HDF when streaming data from disk
    """
    test_file, expected_data = scenarios.chunked_segment().values

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            h5_path = tmp_path / 'h5_streaming_data_test.h5'
            h5 = tdms_file.as_hdf(h5_path)

    for ((group, channel), expected_data) in expected_data.items():
        h5_channel = h5[group][channel]
        np.testing.assert_almost_equal(h5_channel[...], expected_data)
    h5.close()


def test_int_data_types(tmp_path):
    """ Test conversion of signed and unsigned integer types to HDF
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'i8'", 1, 4),
            channel_metadata("/'group'/'u8'", 5, 4),
            channel_metadata("/'group'/'i16'", 2, 4),
            channel_metadata("/'group'/'u16'", 6, 4),
            channel_metadata("/'group'/'i32'", 3, 4),
            channel_metadata("/'group'/'u32'", 7, 4),
            channel_metadata("/'group'/'i64'", 4, 4),
            channel_metadata("/'group'/'u64'", 8, 4),
        ),
        "01 02 03 04"
        "01 02 03 04"
        "01 00 02 00 03 00 04 00"
        "01 00 02 00 03 00 04 00"
        "01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00"
        "01 00 00 00 02 00 00 00 03 00 00 00 04 00 00 00"
        "01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00"
        "01 00 00 00 00 00 00 00 02 00 00 00 00 00 00 00 03 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00"
    )

    tdms_data = test_file.load()
    h5_path = tmp_path / 'h5_data_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    for chan, expected_dtype in [
            ('i8', np.dtype('int8')),
            ('u8', np.dtype('uint8')),
            ('i16', np.dtype('int16')),
            ('u16', np.dtype('uint16')),
            ('i32', np.dtype('int32')),
            ('u32', np.dtype('uint32')),
            ('i64', np.dtype('int64')),
            ('u64', np.dtype('uint64'))]:
        h5_channel = h5['group'][chan]
        assert h5_channel.dtype == expected_dtype
        np.testing.assert_almost_equal(h5_channel[...], [1, 2, 3, 4])
    h5.close()


def test_floating_point_data_types(tmp_path):
    """ Test conversion of f32 and f64 types to HDF
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'f32'", 9, 4),
            channel_metadata("/'group'/'f64'", 10, 4),
        ),
        hexlify_value('<f', 1) +
        hexlify_value('<f', 2) +
        hexlify_value('<f', 3) +
        hexlify_value('<f', 4) +
        hexlify_value('<d', 1) +
        hexlify_value('<d', 2) +
        hexlify_value('<d', 3) +
        hexlify_value('<d', 4)
    )

    tdms_data = test_file.load()
    h5_path = tmp_path / 'h5_data_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    for chan, expected_dtype in [
            ('f32', np.dtype('float32')),
            ('f64', np.dtype('float64'))]:
        h5_channel = h5['group'][chan]
        assert h5_channel.dtype == expected_dtype
        np.testing.assert_almost_equal(h5_channel[...], [1.0, 2.0, 3.0, 4.0])
    h5.close()


def test_timestamp_data(tmp_path):
    """ Test conversion of timestamp channel data to HDF
        HDF doesn't support timestamps natively, so these are converted to strings
    """

    test_file, expected_data = scenarios.timestamp_data().values
    tdms_data = test_file.load()
    h5_path = tmp_path / 'h5_timestamp_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    for (group, channel), expected_values in expected_data.items():
        h5_channel = h5[group][channel]
        assert h5_channel.dtype.kind == 'S'
        expected_strings = np.datetime_as_string(expected_values, unit='us', timezone='UTC')
        expected_ascii = [s.encode('ascii') for s in expected_strings]
        compare_arrays(h5_channel[...], expected_ascii)
    h5.close()


def test_hdf_properties(tmp_path):
    """ Test properties are converted to attributes in HDF files
    """
    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    h5_path = tmp_path / 'h5_properties_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    # File level properties
    assert h5.attrs['num'] == 15

    # Group properties
    assert h5['Group'].attrs['prop'] == 'value'
    assert h5['Group'].attrs['num'] == 10

    # Channel properties
    assert h5['Group']['Channel2'].attrs['wf_start_offset'] == 0.0
    assert h5['Group']['Channel2'].attrs['wf_increment'] == 0.1


def test_timestamp_property(tmp_path):
    """ Test a timestamp property is converted to an attribute in an HDF file
        HDF doesn't support timestamps natively, so these are converted to strings
    """
    test_file = GeneratedFile()
    properties = {
        "wf_start_time": (0x44, hexlify_value("<Q", 0) + hexlify_value("<q", 3524551547))
    }
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2, properties),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    tdms_data = test_file.load()

    h5_path = tmp_path / 'h5_properties_test.h5'
    h5 = tdms_data.as_hdf(h5_path)

    assert h5['group']['channel1'].attrs['wf_start_time'] == b'2015-09-08T10:05:47.000000Z'


def test_as_hdf_string(tmp_path):
    """ Test HDF5 conversion for string datatype
    """
    strings = ["abc123", "?<>~`!@#$%^&*()-=_+,.;'[]:{}|"]

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "11 00 00 00")
    metadata += string_hexlify("/'Group'/'String'")
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
        "2B 00 00 00"
        "00 00 00 00"
        # Number of properties (0)
        "00 00 00 00")
    data = (
        "06 00 00 00"  # index to after first string
        "24 00 00 00"  # index to after second string
    )
    for string in strings:
        data += string_hexlify(string)
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["String"].data
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read

    h5_path = tmp_path / 'h5_strings_test.h5'
    h5 = tdms_data.as_hdf(h5_path)
    h5_strings = h5['Group']['String']
    assert h5_strings.dtype.kind == 'O'
    assert h5_strings.shape[0] == len(strings)
    for expected, read in zip(strings, h5_strings[...]):
        assert expected == _as_string(read)
    h5.close()


def test_unicode_string_data(tmp_path):
    """ Test HDF5 conversion for string datatype with non-ASCII data
    """
    strings = ["Hello, \u4E16\u754C", "\U0001F600"]
    sizes = [len(s.encode('utf-8')) for s in strings]

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    metadata = (
        # Number of objects
        "01 00 00 00"
        # Length of the object path
        "11 00 00 00")
    metadata += string_hexlify("/'Group'/'String'")
    metadata += (
        # Length of index information
        "1C 00 00 00"
        # Raw data data type
        "20 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00" +
        # Number of bytes in data, including index
        hexlify_value('q', sum(sizes) + 4 * len(sizes)) +
        # Number of properties (0)
        "00 00 00 00")
    data = ""
    offset = 0
    for size in sizes:
        # Index gives end positions of strings:
        offset += size
        data += hexlify_value('i', offset)
    for string in strings:
        data += string_hexlify(string)
    test_file.add_segment(toc, metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["String"].data
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read

    h5_path = tmp_path / 'h5_unicode_strings_test.h5'
    h5 = tdms_data.as_hdf(h5_path)
    h5_strings = h5['Group']['String']
    assert h5_strings.dtype.kind == 'O'
    assert h5_strings.shape[0] == len(strings)
    for expected, read in zip(strings, h5_strings[...]):
        assert expected == _as_string(read)
    h5.close()


def test_add_to_file_under_group(tmp_path):
    """ Test adding TDMS data to an HDF file under a group
    """
    test_file, expected_data = scenarios.single_segment_with_two_channels().values
    preexisting_data = np.array([1.0, 2.0, 3.0])

    tdms_data = test_file.load()
    h5_path = tmp_path / 'h5_data_test.h5'
    h5 = h5py.File(h5_path, 'w')
    h5['preexisting_data'] = preexisting_data
    h5.close()

    h5 = tdms_data.as_hdf(h5_path, mode='a', group='tdms_data')

    for ((group, channel), expected_data) in expected_data.items():
        h5_channel = h5['tdms_data'][group][channel]
        np.testing.assert_almost_equal(h5_channel[...], expected_data)
    np.testing.assert_almost_equal(h5['preexisting_data'], preexisting_data)
    h5.close()


def _as_string(bytes_or_string):
    # h5py reads string data as bytes by default since 3.0
    if isinstance(bytes_or_string, bytes):
        return bytes_or_string.decode('UTF-8')
    return bytes_or_string
