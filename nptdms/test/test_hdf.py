""" Test exporting TDMS data to HDF
"""
import pytest
import numpy as np
try:
    import h5py
except ImportError:
    pytest.skip("Skipping HDF tests as h5py is not installed", allow_module_level=True)

from nptdms.test.util import (
    GeneratedFile,
    basic_segment,
    hexlify_value,
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
        np.testing.assert_almost_equal(h5_channel.value, expected_data)
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

    data = tdms_data.channel_data("Group", "String")
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read

    h5_path = tmp_path / 'h5_strings_test.h5'
    h5 = tdms_data.as_hdf(h5_path)
    h5_strings = h5['Group']['String']
    assert h5_strings.dtype.kind == 'S'
    assert h5_strings.shape[0] == len(strings)
    for expected, read in zip(strings, h5_strings.value):
        assert expected == read.decode('utf8')
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

    data = tdms_data.channel_data("Group", "String")
    assert len(data) == len(strings)
    for expected, read in zip(strings, data):
        assert expected == read

    h5_path = tmp_path / 'h5_unicode_strings_test.h5'
    h5 = tdms_data.as_hdf(h5_path)
    h5_strings = h5['Group']['String']
    assert h5_strings.dtype.kind == 'O'
    assert h5_strings.shape[0] == len(strings)
    for expected, read in zip(strings, h5_strings.value):
        assert expected == read
    h5.close()
