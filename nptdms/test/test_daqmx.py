"""Test reading of TDMS files with DAQmx data
"""

import numpy as np

from nptdms import TdmsFile
from nptdms.test.util import (
    GeneratedFile, hexlify_value, string_hexlify, segment_objects_metadata, hex_properties)


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

    data = tdms_data.object("Group", "Channel1").raw_data

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

    data = tdms_data.object("Group", "Channel1").raw_data

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

    data = tdms_data.object("Group", "Channel1").raw_data

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

    data = tdms_data.object("Group", "Channel1").raw_data

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

    data_1 = tdms_data.object("Group", "Channel1").raw_data
    assert data_1.dtype == np.int16
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data.object("Group", "Channel2").raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])


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

    data_1 = tdms_data.object("Group", "Channel1").raw_data
    assert data_1.dtype == np.int8
    np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

    data_2 = tdms_data.object("Group", "Channel2").raw_data
    assert data_2.dtype == np.int16
    np.testing.assert_array_equal(data_2, [17, 18, 19, 20])

    data_3 = tdms_data.object("Group", "Channel3").raw_data
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
    channel = tdms_data.object("Group", "Channel1")

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
    channel = tdms_data.object("Group", "Channel1")

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

    data_1 = tdms_data.object("Group", "Channel1").raw_data
    data_2 = tdms_data.object("Group", "Channel2").raw_data
    data_3 = tdms_data.object("Group", "Channel3").raw_data
    data_4 = tdms_data.object("Group", "Channel4").raw_data

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

    data_1 = tdms_data.object("Group", "Channel1").raw_data
    data_2 = tdms_data.object("Group", "Channel2").raw_data
    data_3 = tdms_data.object("Group", "Channel3").raw_data
    data_4 = tdms_data.object("Group", "Channel4").raw_data
    data_5 = tdms_data.object("Group", "Channel5").raw_data

    for data in [data_1, data_2, data_3]:
        assert data.dtype == np.int16
    for data in [data_4, data_5]:
        assert data.dtype == np.int32

    np.testing.assert_array_equal(data_1, [1, 4, 7, 10])
    np.testing.assert_array_equal(data_2, [2, 5, 8, 11])
    np.testing.assert_array_equal(data_3, [3, 6, 9, 12])
    np.testing.assert_array_equal(data_4, [13, 15, 17, 19])
    np.testing.assert_array_equal(data_5, [14, 16, 18, 20])


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

    channel_1 = tdms_data.object("Group", "Channel1")
    channel_2 = tdms_data.object("Group", "Channel2")

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
            data_1 = tdms_file.object("Group", "Channel1").read_data()
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [1, 2, 3, 4])

            data_2 = tdms_file.object("Group", "Channel2").read_data()
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
            data_1 = tdms_file.object("Group", "Channel1").read_data(1, 2)
            assert data_1.dtype == np.int16
            np.testing.assert_array_equal(data_1, [2, 3])

            data_2 = tdms_file.object("Group", "Channel2").read_data(1, 2)
            assert data_2.dtype == np.int16
            np.testing.assert_array_equal(data_2, [18, 19])


def segment_toc():
    return (
        "kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocDAQmxRawData")


def root_metadata():
    return (
        # Length of the object path
        "01 00 00 00"
        # Object path (/)
        "2F"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00")


def group_metadata():
    return (
        # Length of the object path
        "08 00 00 00"
        # Object path (/'Group')
        "2F 27 47 72"
        "6F 75 70 27"
        # Raw data index
        "FF FF FF FF"
        # Num properties
        "00 00 00 00")


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


def daqmx_channel_metadata(
        channel_name, num_values,
        raw_data_widths, scaler_metadata, properties=None):
    path = "/'Group'/'" + channel_name + "'"
    return (
        # Length of the object path
        hexlify_value("<I", len(path)) +
        # Object path
        string_hexlify(path) +
        # Raw data index (DAQmx)
        "69 12 00 00"
        # Data type (DAQmx)
        "FF FF FF FF"
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
