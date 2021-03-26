"""Test reading of TDMS files with DAQmx data
"""

from collections import defaultdict
import logging
import numpy as np
import pytest

from nptdms import TdmsFile
from nptdms import types
from nptdms.log import log_manager
from nptdms.test.scenarios import timestamp_hexlify
from nptdms.test.util import (
    GeneratedFile, hexlify_value, string_hexlify, segment_objects_metadata, hex_properties,
    root_metadata, group_metadata)


def test_single_channel_i16():
    """ Test loading a DAQmx file with a single channel of I16 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 3, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.int16
    np.testing.assert_array_equal(data, [1, 2, -1, -2])


def test_single_channel_u16():
    """ Test loading a DAQmx file with a single channel of U16 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint16
    np.testing.assert_array_equal(data, [1, 2, 2**16 - 1, 2**16 - 2])


def test_single_channel_i32():
    """ Test loading a DAQmx file with a single channel of I32 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 5, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "FF FF FF FF"
        "FE FF FF FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.int32
    np.testing.assert_array_equal(data, [1, 2, -1, -2])


def test_single_channel_u32():
    """ Test loading a DAQmx file with a single channel of U32 data
    """

    scaler_metadata = daqmx_scaler_metadata(0, 4, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata]))
    data = (
        # Data for segment
        "01 00 00 00"
        "02 00 00 00"
        "FF FF FF FF"
        "FE FF FF FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint32
    np.testing.assert_array_equal(data, [1, 2, 2**32 - 1, 2**32 - 2])


def test_two_channel_i16():
    """ Test loading a DAQmx file with two channels of I16 data
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2]))
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
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int16
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_two_channels_without_daqmx_toc_flag():
    """ Test loading a DAQmx file with two channels of DAQmx data but where
        the segment header is missing the DAQmx flag (see issue #226)
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2]))
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
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int16
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_daqmx_metadata_without_daqmx_raw_data():
    """ Test loading a file that uses DAQmx format metadata but where the data types are not raw data (see issue #226)
    """

    scaler_1 = daqmx_scaler_metadata(0, 0xFFFFFFFF, 0)  # Timestamp data
    scaler_2 = daqmx_scaler_metadata(0, 7, 16)
    scaler_3 = daqmx_scaler_metadata(0, 9, 24)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [32], [scaler_1], data_type=types.TimeStamp.enum_value),
        daqmx_channel_metadata("Channel2", 4, [32], [scaler_2], data_type=types.Int64.enum_value),
        daqmx_channel_metadata("Channel3", 4, [32], [scaler_3], data_type=types.DoubleFloat.enum_value))

    times = [
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]

    data = (
        timestamp_hexlify(times[0]) +
        "01 00 00 00 00 00 00 00" +
        hexlify_value("<d", 1.0) +
        timestamp_hexlify(times[1]) +
        "02 00 00 00 00 00 00 00" +
        hexlify_value("<d", 2.0) +
        timestamp_hexlify(times[2]) +
        "03 00 00 00 00 00 00 00" +
        hexlify_value("<d", 3.0) +
        timestamp_hexlify(times[3]) +
        "04 00 00 00 00 00 00 00" +
        hexlify_value("<d", 4.0)
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"][:]
    np.testing.assert_array_equal(data_1, times)

    data_2 = tdms_data["Group"]["Channel2"][:]
    assert data_2.dtype == np.int64
    np.testing.assert_array_equal(data_2, [1, 2, 3, 4])

    data_3 = tdms_data["Group"]["Channel3"][:]
    assert data_3.dtype == np.float64
    np.testing.assert_array_equal(data_3, [1.0, 2.0, 3.0, 4.0])


def test_exception_on_mismatch_of_types_for_non_raw_daqmx():
    scaler = daqmx_scaler_metadata(0, 3, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 2, [2], [scaler], data_type=types.Uint16.enum_value))
    data = "01 00 02 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert error_message == "Expected scaler data type to be Uint16 but got Int16"


def test_exception_on_multiple_scalers_for_non_raw_daqmx():
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 2, [4], [scaler_1, scaler_2], data_type=types.Int32.enum_value))
    data = "01 00 00 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert error_message == "Expected only one scaler for channel with type Int32"


def test_mixed_channel_widths():
    """ Test loading a DAQmx file with channels with different widths
    """

    scaler_1 = daqmx_scaler_metadata(0, 1, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 1)
    scaler_3 = daqmx_scaler_metadata(0, 5, 3)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [7], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [7], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [7], [scaler_3]))
    data = (
        # Data for segment
        "01 11 00 21 00 00 00"
        "02 12 00 22 00 00 00"
        "03 13 00 23 00 00 00"
        "04 14 00 24 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    assert data_1.dtype == np.int8
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data["Group"]["Channel2"].raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])

    data_3 = tdms_data["Group"]["Channel3"].raw_data
    assert data_3.dtype == np.int32
    np.testing.assert_array_equal(data_3, [33, 34, 35, 36])


def test_multiple_scalers_with_same_type():
    """ Test loading a DAQmx file with one channel containing multiple
        format changing scalers of the same type
    """

    scaler_metadata = [
        daqmx_scaler_metadata(0, 3, 0),
        daqmx_scaler_metadata(1, 3, 2)]
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
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
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]

    scaler_0_data = channel.raw_scaler_data[0]
    assert scaler_0_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_0_data, [1, 2, 3, 4])

    scaler_1_data = channel.raw_scaler_data[1]
    assert scaler_1_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_1_data, [17, 18, 19, 20])


def test_multiple_scalers_with_different_types():
    """ Test loading a DAQmx file with one channel containing multiple
        format changing scalers of different types
    """

    scaler_metadata = [
        daqmx_scaler_metadata(0, 1, 0),
        daqmx_scaler_metadata(1, 3, 1),
        daqmx_scaler_metadata(2, 5, 3)]
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [7], scaler_metadata))
    data = (
        # Data for segment
        "01 11 00 21 00 00 00"
        "02 12 00 22 00 00 00"
        "03 13 00 23 00 00 00"
        "04 14 00 24 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()
    channel = tdms_data["Group"]["Channel1"]

    scaler_0_data = channel.raw_scaler_data[0]
    assert scaler_0_data.dtype == np.int8
    np.testing.assert_array_equal(scaler_0_data, [1, 2, 3, 4])

    scaler_1_data = channel.raw_scaler_data[1]
    assert scaler_1_data.dtype == np.int16
    np.testing.assert_array_equal(scaler_1_data, [17, 18, 19, 20])

    scaler_2_data = channel.raw_scaler_data[2]
    assert scaler_2_data.dtype == np.int32
    np.testing.assert_array_equal(scaler_2_data, [33, 34, 35, 36])


def test_multiple_raw_data_buffers():
    """ Test loading a DAQmx file with multiple raw data buffers
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0, 3, 2, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4, 4], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [4, 4], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [4, 4], [scaler_3]),
        daqmx_channel_metadata("Channel4", 4, [4, 4], [scaler_4]))
    data = (
        "01 00" "02 00" "03 00" "04 00"
        "05 00" "06 00" "07 00" "08 00"
        "09 00" "0A 00" "0B 00" "0C 00"
        "0D 00" "0E 00" "0F 00" "10 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    data_2 = tdms_data["Group"]["Channel2"].raw_data
    data_3 = tdms_data["Group"]["Channel3"].raw_data
    data_4 = tdms_data["Group"]["Channel4"].raw_data

    for data in [data_1, data_2, data_3, data_4]:
        assert data.dtype == np.int16

    np.testing.assert_array_equal(data_1, [1, 3, 5, 7])
    np.testing.assert_array_equal(data_2, [2, 4, 6, 8])
    np.testing.assert_array_equal(data_3, [9, 11, 13, 15])
    np.testing.assert_array_equal(data_4, [10, 12, 14, 16])


def test_multiple_raw_data_buffers_with_different_widths():
    """ DAQmx with raw data buffers with different widths
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 4, 0)
    scaler_4 = daqmx_scaler_metadata(0, 5, 0, 1)
    scaler_5 = daqmx_scaler_metadata(0, 5, 4, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [6, 8], [scaler_1]),
        daqmx_channel_metadata("Channel2", 4, [6, 8], [scaler_2]),
        daqmx_channel_metadata("Channel3", 4, [6, 8], [scaler_3]),
        daqmx_channel_metadata("Channel4", 4, [6, 8], [scaler_4]),
        daqmx_channel_metadata("Channel5", 4, [6, 8], [scaler_5]))
    data = (
        "01 00" "02 00" "03 00"
        "04 00" "05 00" "06 00"
        "07 00" "08 00" "09 00"
        "0A 00" "0B 00" "0C 00"
        "0D 00 00 00" "0E 00 00 00"
        "0F 00 00 00" "10 00 00 00"
        "11 00 00 00" "12 00 00 00"
        "13 00 00 00" "14 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"].raw_data
    data_2 = tdms_data["Group"]["Channel2"].raw_data
    data_3 = tdms_data["Group"]["Channel3"].raw_data
    data_4 = tdms_data["Group"]["Channel4"].raw_data
    data_5 = tdms_data["Group"]["Channel5"].raw_data

    for data in [data_1, data_2, data_3]:
        assert data.dtype == np.int16
    for data in [data_4, data_5]:
        assert data.dtype == np.int32

    np.testing.assert_array_equal(data_1, [1, 4, 7, 10])
    np.testing.assert_array_equal(data_2, [2, 5, 8, 11])
    np.testing.assert_array_equal(data_3, [3, 6, 9, 12])
    np.testing.assert_array_equal(data_4, [13, 15, 17, 19])
    np.testing.assert_array_equal(data_5, [14, 16, 18, 20])


def test_multiple_raw_data_buffers_with_different_lengths():
    """ DAQmx with raw data buffers with different lengths
    """

    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 5, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0, 3, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0, 5, 2, 1)
    scaler_5 = daqmx_scaler_metadata(0, 3, 0, 2)
    scaler_6 = daqmx_scaler_metadata(0, 5, 2, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [6, 6, 6], [scaler_1], properties=properties),
        daqmx_channel_metadata("Channel2", 4, [6, 6, 6], [scaler_2], properties=properties),
        daqmx_channel_metadata("Channel3", 2, [6, 6, 6], [scaler_3], properties=properties),
        daqmx_channel_metadata("Channel4", 2, [6, 6, 6], [scaler_4], properties=properties),
        daqmx_channel_metadata("Channel5", 1, [6, 6, 6], [scaler_5], properties=properties),
        daqmx_channel_metadata("Channel6", 1, [6, 6, 6], [scaler_6], properties=properties))
    data = (
        # Chunk 1
        # Buffer 0
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        # Buffer 1
        "03 00" "04 00 00 00"
        "03 00" "04 00 00 00"
        # Buffer 2
        "05 00" "06 00 00 00"
        # Chunk 2
        # Buffer 0
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        "01 00" "02 00 00 00"
        # Buffer 1
        "03 00" "04 00 00 00"
        "03 00" "04 00 00 00"
        # Buffer 2
        "05 00" "06 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data_1 = tdms_data["Group"]["Channel1"][:]
    data_2 = tdms_data["Group"]["Channel2"][:]
    data_3 = tdms_data["Group"]["Channel3"][:]
    data_4 = tdms_data["Group"]["Channel4"][:]
    data_5 = tdms_data["Group"]["Channel5"][:]
    data_6 = tdms_data["Group"]["Channel6"][:]

    for data in [data_1, data_3, data_5]:
        assert data.dtype == np.int16
    for data in [data_2, data_4, data_6]:
        assert data.dtype == np.int32

    np.testing.assert_array_equal(data_1, [1] * 8)
    np.testing.assert_array_equal(data_2, [2] * 8)
    np.testing.assert_array_equal(data_3, [3] * 4)
    np.testing.assert_array_equal(data_4, [4] * 4)
    np.testing.assert_array_equal(data_5, [5] * 2)
    np.testing.assert_array_equal(data_6, [6] * 2)


def test_incomplete_segment_with_different_length_buffers():
    """ DAQmx with raw data buffers with different lengths
    """

    scaler_1 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 0)
    scaler_3 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 1)
    scaler_4 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 1)
    scaler_5 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 0, 2)
    scaler_6 = daqmx_scaler_metadata(0xFFFFFFFF, 2, 2, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4, 4, 4], [scaler_1], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel2", 4, [4, 4, 4], [scaler_2], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel3", 2, [4, 4, 4], [scaler_3], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel4", 2, [4, 4, 4], [scaler_4], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel5", 1, [4, 4, 4], [scaler_5], data_type=types.Uint16.enum_value),
        daqmx_channel_metadata("Channel6", 1, [4, 4, 4], [scaler_6], data_type=types.Uint16.enum_value))
    data = (
        # Chunk 1
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
        "03 00" "04 00"
        # Buffer 2
        "05 00" "06 00"
        # Chunk 2
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
        "03 00" "04 00"
        # Buffer 2
        "05 00" "06 00"
        # Incomplete third chunk
        # Buffer 0
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        "01 00" "02 00"
        # Buffer 1
        "03 00" "04 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc_non_daqmx(), metadata, data, incomplete=True)
    tdms_data = test_file.load()

    group = tdms_data["Group"]

    np.testing.assert_array_equal(group["Channel1"][:], [1] * 12)
    np.testing.assert_array_equal(group["Channel2"][:], [2] * 12)
    np.testing.assert_array_equal(group["Channel3"][:], [3] * 5)
    np.testing.assert_array_equal(group["Channel4"][:], [4] * 5)
    np.testing.assert_array_equal(group["Channel5"][:], [5] * 2)
    np.testing.assert_array_equal(group["Channel6"][:], [6] * 2)


def test_multiple_raw_data_buffers_with_scalers_split_across_buffers():
    """ DAQmx with scalers split across different raw data buffers
    """

    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(1, 3, 0, 1)
    scaler_3 = daqmx_scaler_metadata(0, 3, 2, 0)
    scaler_4 = daqmx_scaler_metadata(1, 3, 2, 1)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata(
            "Channel1", 4, [4, 4], [scaler_1, scaler_2]),
        daqmx_channel_metadata(
            "Channel2", 4, [4, 4], [scaler_3, scaler_4]))
    data = (
        "01 00" "02 00" "03 00" "04 00"
        "05 00" "06 00" "07 00" "08 00"
        "09 00" "0A 00" "0B 00" "0C 00"
        "0D 00" "0E 00" "0F 00" "10 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    channel_1 = tdms_data["Group"]["Channel1"]
    channel_2 = tdms_data["Group"]["Channel2"]

    scaler_data_1 = channel_1.raw_scaler_data[0]
    scaler_data_2 = channel_1.raw_scaler_data[1]
    scaler_data_3 = channel_2.raw_scaler_data[0]
    scaler_data_4 = channel_2.raw_scaler_data[1]

    for data in [
            scaler_data_1, scaler_data_2, scaler_data_3, scaler_data_4]:
        assert data.dtype == np.int16

    np.testing.assert_array_equal(scaler_data_1, [1, 3, 5, 7])
    np.testing.assert_array_equal(scaler_data_2, [9, 11, 13, 15])
    np.testing.assert_array_equal(scaler_data_3, [2, 4, 6, 8])
    np.testing.assert_array_equal(scaler_data_4, [10, 12, 14, 16])


def test_digital_line_scaler_data():
    """ Test loading a DAQmx file with a single channel of U8 digital line scaler data
    """

    scaler_metadata = digital_scaler_metadata(0, 0, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata], digital_line_scaler=True))
    data = (
        "00 00 00 00"
        "01 00 00 00"
        "00 00 00 00"
        "01 00 00 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    tdms_data = test_file.load()

    data = tdms_data["Group"]["Channel1"].raw_data

    assert data.dtype == np.uint8
    np.testing.assert_array_equal(data, [0, 1, 0, 1])


@pytest.mark.parametrize('byte_offset', [0, 1, 2, 3])
def test_digital_line_scaler_with_multiple_channels(byte_offset):
    """ Test DAQmx digital line scaler data with multiple channels
    """

    scaler_metadata_0 = digital_scaler_metadata(0, 0, byte_offset * 8 + 0)
    scaler_metadata_1 = digital_scaler_metadata(0, 0, byte_offset * 8 + 1)
    scaler_metadata_2 = digital_scaler_metadata(0, 0, byte_offset * 8 + 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel0", 4, [4], [scaler_metadata_0], digital_line_scaler=True),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_metadata_1], digital_line_scaler=True),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_metadata_2], digital_line_scaler=True),
    )
    byte_values = [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
    ]
    hex_data = " ".join("00" * byte_offset + b + "00" * (3 - byte_offset) for b in byte_values)

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, hex_data)
    tdms_data = test_file.load()

    for (channel_name, expected_data) in [
        ("Channel0", [0, 1, 0, 1, 0, 1, 0, 1]),
        ("Channel1", [0, 0, 1, 1, 0, 0, 1, 1]),
        ("Channel2", [0, 0, 0, 0, 1, 1, 1, 1]),
    ]:
        data = tdms_data["Group"][channel_name].raw_data

        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, expected_data, "Incorrect data for channel '%s'" % channel_name)


def test_exception_on_mismatch_of_raw_data_widths():
    scaler_1 = daqmx_scaler_metadata(0, 3, 0, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 1, [2], [scaler_1]),
        daqmx_channel_metadata("Channel2", 1, [4], [scaler_2]))
    data = "01 00 02 00"

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    with pytest.raises(ValueError) as exception:
        _ = test_file.load()
    error_message = str(exception.value)

    assert (error_message == "Raw data widths for object DaqmxSegmentObject(/'Group'/'Channel2') ([4]) "
            "do not match previous widths ([2])")


def test_lazily_reading_channel():
    """ Test loading channels individually from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
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
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data()
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

            data_2 = tdms_file["Group"]["Channel2"].read_data()
            assert data_2.dtype == np.int16
            np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


def test_lazily_reading_a_subset_of_channel_data():
    """ Test loading a subset of channel data from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
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
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data(1, 2)
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [2, 3])

            data_2 = tdms_file["Group"]["Channel2"].read_data(1, 2)
            assert data_2.dtype == np.int16
            np.testing.assert_array_equal(data_2, [18, 19])


def test_lazily_reading_a_subset_of_raw_channel_data():
    """ Test loading a subset of raw scaler channel data from a DAQmx file
    """

    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
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
    test_file.add_segment(segment_toc(), metadata, data)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            data_1 = tdms_file["Group"]["Channel1"].read_data(1, 2, scaled=False)
            assert len(data_1) == 1
            assert data_1[0].dtype == np.int16
            np.testing.assert_array_equal(data_1[0], [2, 3])

            data_2 = tdms_file["Group"]["Channel2"].read_data(1, 2, scaled=False)
            assert len(data_2) == 1
            assert data_2[0].dtype == np.int16
            np.testing.assert_array_equal(data_2[0], [18, 19])


@pytest.mark.parametrize('offset,length', [
    (0, None),
    (1, None),
    (0, 2),
    (1, 2),
])
def test_read_raw_data(offset, length):
    # Single scale which is just the raw DAQmx scaler data
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(1, 3, 0)
    scaler_2 = daqmx_scaler_metadata(2, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
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
    test_file.add_segment(segment_toc(), metadata, data)

    end = None if length is None else offset + length
    with test_file.get_tempfile() as temp_file:
        tdms_file = TdmsFile.read(temp_file.file)
        data_1 = tdms_file["Group"]["Channel1"].read_data(offset=offset, length=length, scaled=False)
        assert data_1[1].dtype == np.int16
        np.testing.assert_array_equal(data_1[1], [1, 2, 3, 4][offset:end])

        data_2 = tdms_file["Group"]["Channel2"].read_data(offset=offset, length=length, scaled=False)
        assert data_2[2].dtype == np.int16
        np.testing.assert_array_equal(data_2[2], [17, 18, 19, 20][offset:end])


def test_stream_data_chunks():
    """Test streaming chunks of DAQmx data from a TDMS file
    """
    properties = {
        "NI_Number_Of_Scales": (3, "01 00 00 00"),
    }
    scaler_1 = daqmx_scaler_metadata(0, 3, 0)
    scaler_2 = daqmx_scaler_metadata(0, 3, 2)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [4], [scaler_1], properties),
        daqmx_channel_metadata("Channel2", 4, [4], [scaler_2], properties))
    data = (
        # Data for segment
        "01 00" "11 00"
        "02 00" "12 00"
        "03 00" "13 00"
        "04 00" "14 00"
        "05 00" "15 00"
        "06 00" "16 00"
        "07 00" "17 00"
        "08 00" "18 00"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)
    data_arrays = defaultdict(list)
    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            for chunk in tdms_file.data_chunks():
                for group in chunk.groups():
                    for channel in group.channels():
                        key = (group.name, channel.name)
                        assert channel.offset == len(data_arrays[key])
                        data_arrays[key].extend(channel[:])

    expected_channel_data = {
        ("Group", "Channel1"): [1, 2, 3, 4, 5, 6, 7, 8],
        ("Group", "Channel2"): [17, 18, 19, 20, 21, 22, 23, 24],
    }
    for ((group, channel), expected_data) in expected_channel_data.items():
        actual_data = data_arrays[(group, channel)]
        np.testing.assert_equal(actual_data, expected_data)


def test_daqmx_debug_logging(caplog):
    """ Test loading a DAQmx file with debug logging enabled
    """
    scaler_metadata = daqmx_scaler_metadata(0, 3, 0)
    metadata = segment_objects_metadata(
        root_metadata(),
        group_metadata(),
        daqmx_channel_metadata("Channel1", 4, [2], [scaler_metadata]))
    data = (
        "01 00"
        "02 00"
        "FF FF"
        "FE FF"
    )

    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, data)

    log_manager.set_level(logging.DEBUG)
    _ = test_file.load()

    assert "Reading metadata for object /'Group'/'Channel1' with index header 0x00001269" in caplog.text
    assert "scale_id=0" in caplog.text
    assert "data_type=Int16" in caplog.text


def segment_toc():
    return (
        "kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocDAQmxRawData")


def segment_toc_non_daqmx():
    return (
        "kTocMetaData", "kTocRawData", "kTocNewObjList")


def daqmx_scaler_metadata(scale_id, type_id, byte_offset, raw_buffer_index=0):
    return (
        # DAQmx data type (type ids don't match TDMS types)
        hexlify_value("<I", type_id) +
        # Raw buffer index
        hexlify_value("<I", raw_buffer_index) +
        # Raw byte offset
        hexlify_value("<I", byte_offset) +
        # Sample format bitmap (don't know what this is for...)
        "00 00 00 00" +
        # Scale ID
        hexlify_value("<I", scale_id))


def digital_scaler_metadata(scale_id, type_id, bit_offset, raw_buffer_index=0):
    return (
        # DAQmx data type (type ids don't match TDMS types)
        hexlify_value("<I", type_id) +
        # Raw buffer index
        hexlify_value("<I", raw_buffer_index) +
        # Raw byte offset
        hexlify_value("<I", bit_offset) +
        # Sample format bitmap (don't know what this is for...)
        "00" +
        # Scale ID
        hexlify_value("<I", scale_id))


def daqmx_channel_metadata(
        channel_name, num_values,
        raw_data_widths, scaler_metadata, properties=None, digital_line_scaler=False,
        data_type=None):
    path = "/'Group'/'" + channel_name + "'"
    # Default to DAQmx data type
    data_type = 0xFFFFFFFF if data_type is None else data_type
    return (
        # Length of the object path
        hexlify_value("<I", len(path)) +
        # Object path
        string_hexlify(path) +
        # Raw data index (DAQmx)
        ("6A 12 00 00" if digital_line_scaler else "69 12 00 00") +
        # Data type
        hexlify_value("<I", data_type) +
        # Array  dimension
        "01 00 00 00" +
        # Number of values (chunk size)
        hexlify_value("<Q", num_values) +
        # Scaler metadata
        hexlify_value("<I", len(scaler_metadata)) +
        "".join(scaler_metadata) +
        # Raw data width vector size
        hexlify_value("<I", len(raw_data_widths)) +
        # Raw data width values
        "".join(hexlify_value("<I", v) for v in raw_data_widths) +
        hex_properties(properties))
