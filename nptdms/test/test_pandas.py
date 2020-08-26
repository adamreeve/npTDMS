"""Test exporting TDMS data to Pandas"""

from datetime import datetime
import numpy as np
import pytest
try:
    import pandas
except ImportError:
    pytest.skip("Skipping Pandas tests as Pandas is not installed", allow_module_level=True)

from nptdms import TdmsFile
from nptdms.test import scenarios
from nptdms.test.test_daqmx import daqmx_channel_metadata, daqmx_scaler_metadata
from nptdms.test.util import (
    GeneratedFile,
    basic_segment,
    channel_metadata,
    channel_metadata_with_no_data,
    string_hexlify,
    segment_objects_metadata,
    hexlify_value
)


def assert_within_tol(a, b, tol=1.0e-10):
    assert abs(a - b) < tol


def timed_segment():
    """TDMS segment with one group and two channels,
    each with time properties"""

    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
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
        "02 00 00 00"
        # Name length
        "04 00 00 00"
        # Property name (prop)
        "70 72 6F 70"
        # Property data type (string)
        "20 00 00 00"
        # Length of string value
        "05 00 00 00"
        # Value
        "76 61 6C 75 65"
        # Length of second property name
        "03 00 00 00"
        # Property name (num)
        "6E 75 6D"
        # Data type of property
        "03 00 00 00"
        # Value
        "0A 00 00 00"
        # Length of the second object path
        "13 00 00 00"
        # Second object path (/'Group'/'Channel1')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 31 27"
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of raw data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties
        "03 00 00 00"
        # Set time properties for the first channel
        "0F 00 00 00" +
        string_hexlify('wf_start_offset') +
        "0A 00 00 00" +
        hexlify_value("<d", 2.0) +
        "0C 00 00 00" +
        string_hexlify('wf_increment') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.1) +
        "0D 00 00 00" +
        string_hexlify('wf_start_time') +
        "44 00 00 00" +
        hexlify_value("<Q", 0) +
        hexlify_value("<q", 3524551547) +
        # Length of the third object path
        "13 00 00 00"
        # Third object path (/'Group'/'Channel2')
        "2F 27 47 72"
        "6F 75 70 27"
        "2F 27 43 68"
        "61 6E 6E 65"
        "6C 32 27"
        # Length of index information
        "14 00 00 00"
        # Raw data data type
        "03 00 00 00"
        # Dimension
        "01 00 00 00"
        # Number of data values
        "02 00 00 00"
        "00 00 00 00"
        # Number of properties
        "03 00 00 00"
        # Set time properties for the second channel
        "0F 00 00 00" +
        string_hexlify('wf_start_offset') +
        "0A 00 00 00" +
        hexlify_value("<d", 2.0) +
        "0C 00 00 00" +
        string_hexlify('wf_increment') +
        "0A 00 00 00" +
        hexlify_value("<d", 0.1) +
        "0D 00 00 00" +
        string_hexlify('wf_start_time') +
        "44 00 00 00" +
        hexlify_value("<Q", 0) +
        hexlify_value("<q", 3524551547))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "03 00 00 00"
        "04 00 00 00"
    )
    return toc, metadata, data


@pytest.mark.parametrize('lazy_load', [True, False])
def test_file_as_dataframe(lazy_load):
    """Test converting file to Pandas dataframe"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())

    df = on_test_file(test_file, lazy_load, lambda tdms_data: tdms_data.as_dataframe())

    assert len(df) == 2
    assert "/'Group'/'Channel1'" in df.keys()
    assert "/'Group'/'Channel2'" in df.keys()

    assert (df["/'Group'/'Channel1'"] == [1, 2]).all()


def test_file_as_dataframe_without_time():
    """Converting file to dataframe with time index should raise when
    time properties aren't present"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    with pytest.raises(KeyError):
        tdms_data.as_dataframe(time_index=True)


def test_file_as_dataframe_with_time():
    """Test converting file to Pandas dataframe with a time index"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())
    tdms_data = test_file.load()

    df = tdms_data.as_dataframe(time_index=True)

    assert len(df.index) == 2
    assert_within_tol(df.index[0], 2.0)
    assert_within_tol(df.index[1], 2.1)


def test_file_as_dataframe_with_absolute_time():
    """Convert file to Pandas dataframe with absolute time index"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())
    tdms_data = test_file.load()

    df = tdms_data.as_dataframe(time_index=True, absolute_time=True)

    expected_start = datetime(2015, 9, 8, 10, 5, 49)
    assert (df.index == expected_start)[0]


@pytest.mark.parametrize('lazy_load', [True, False])
def test_group_as_dataframe(lazy_load):
    """Convert a group to dataframe"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())

    df = on_test_file(test_file, lazy_load, lambda tdms_data: tdms_data["Group"].as_dataframe())

    assert len(df) == 2
    assert len(df.keys()) == 2
    assert "Channel1" in df.keys()
    assert "Channel2" in df.keys()
    assert (df["Channel1"] == [1, 2]).all()
    assert (df["Channel2"] == [3, 4]).all()


@pytest.mark.parametrize('lazy_load', [True, False])
def test_channel_as_dataframe(lazy_load):
    """Convert a channel to dataframe"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())

    df = on_test_file(
        test_file, lazy_load, lambda tdms_data: tdms_data["Group"]["Channel2"].as_dataframe())

    assert len(df) == 2
    assert len(df.keys()) == 1
    assert "/'Group'/'Channel2'" in df.keys()
    assert (df["/'Group'/'Channel2'"] == [3, 4]).all()


def test_channel_as_dataframe_with_time():
    """Convert a channel to dataframe with a time index"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())
    tdms_data = test_file.load()

    df = tdms_data["Group"]["Channel2"].as_dataframe(time_index=True)

    assert len(df.index) == 2
    assert_within_tol(df.index[0], 2.0)
    assert_within_tol(df.index[1], 2.1)


def test_channel_as_dataframe_without_time():
    """Converting channel to dataframe should work correctly"""

    test_file = GeneratedFile()
    test_file.add_segment(*basic_segment())
    tdms_data = test_file.load()

    df = tdms_data["Group"]["Channel2"].as_dataframe()

    assert len(df.index) == 2
    assert len(df.values) == 2
    assert_within_tol(df.index[0], 0)
    assert_within_tol(df.index[1], 1)
    assert_within_tol(df.values[0], 3.0)
    assert_within_tol(df.values[1], 4.0)


def test_channel_as_dataframe_with_absolute_time():
    """Convert channel to Pandas dataframe with absolute time index"""

    test_file = GeneratedFile()
    test_file.add_segment(*timed_segment())
    tdms_data = test_file.load()

    df = tdms_data["Group"]["Channel1"].as_dataframe(time_index=True, absolute_time=True)

    expected_start = datetime(2015, 9, 8, 10, 5, 49)
    assert (df.index == expected_start)[0]


@pytest.mark.parametrize('lazy_load', [True, False])
def test_channel_as_dataframe_with_raw_data(lazy_load):
    """Convert channel to Pandas dataframe with absolute time index"""

    test_file, _ = scenarios.scaled_data().values
    expected_raw_data = np.array([1, 2, 3, 4], dtype=np.int32)

    df = on_test_file(
        test_file, lazy_load, lambda tdms_data: tdms_data["group"]["channel1"].as_dataframe(scaled_data=False))

    np.testing.assert_equal(df["/'group'/'channel1'"], expected_raw_data)


@pytest.mark.parametrize('lazy_load', [True, False])
def test_raw_daqmx_channel_export(lazy_load):
    """ Test exporting raw daqmx data for a channel
    """

    scaler_metadata = [
        daqmx_scaler_metadata(0, 3, 0),
        daqmx_scaler_metadata(1, 3, 2)]
    metadata = segment_objects_metadata(
        daqmx_channel_metadata("Channel1", 4, [4], scaler_metadata))
    data = (
        # Data for segment
        "01 00"
        "11 00"
        "02 00"
        "12 00"
        "03 00"
        "13 00"
        "04 00"
        "14 00"
    )

    test_file = GeneratedFile()
    segment_toc = (
        "kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocDAQmxRawData")
    test_file.add_segment(segment_toc, metadata, data)
    dataframe = on_test_file(
        test_file, lazy_load, lambda tdms_data: tdms_data["Group"]["Channel1"].as_dataframe(scaled_data=False))

    expected_data = {
        0: np.array([1, 2, 3, 4], dtype=np.int16),
        1: np.array([17, 18, 19, 20], dtype=np.int16),
    }
    assert dataframe["/'Group'/'Channel1'[0]"].dtype == np.int16
    assert dataframe["/'Group'/'Channel1'[1]"].dtype == np.int16
    np.testing.assert_equal(dataframe["/'Group'/'Channel1'[0]"], expected_data[0])
    np.testing.assert_equal(dataframe["/'Group'/'Channel1'[1]"], expected_data[1])


def test_export_with_empty_channels():
    """Convert a group to dataframe when a channel has empty data and void data type"""

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 3, 2),
            channel_metadata_with_no_data("/'group'/'channel2'"),
        ),
        "01 00 00 00 02 00 00 00"
    )

    tdms_data = test_file.load()

    df = tdms_data["group"].as_dataframe()
    assert len(df) == 2
    assert len(df.keys()) == 2
    assert "channel1" in df.keys()
    assert "channel2" in df.keys()
    assert (df["channel1"] == [1, 2]).all()
    assert len(df["channel2"]) == 2
    assert np.isnan(df["channel2"]).all()


def on_test_file(test_file, lazy_load, func):
    if lazy_load:
        with test_file.get_tempfile() as temp_file:
            with TdmsFile.open(temp_file) as tdms_file:
                return func(tdms_file)
    else:
        return func(test_file.load())
