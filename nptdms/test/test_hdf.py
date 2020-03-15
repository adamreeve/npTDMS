"""Test exporting TDMS data to HDF"""

import pytest
try:
    import h5py
except ImportError:
    pytest.skip("Skipping HDF tests as h5py is not installed", allow_module_level=True)

from nptdms.test.util import (
    GeneratedFile,
    string_hexlify
)


def test_as_hdf_string(tmp_path):
    """Test HDF5 conversion for string datatype
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
