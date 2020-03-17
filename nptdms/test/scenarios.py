""" Contains different test cases for tests for reading TDMS files
"""
import numpy as np
import pytest

from nptdms.test.util import (
    GeneratedFile,
    hexlify_value,
    string_hexlify,
    segment_objects_metadata,
    channel_metadata,
    channel_metadata_with_no_data,
    channel_metadata_with_repeated_structure)


TDS_TYPE_INT32 = 3
TDS_TYPE_COMPLEX64 = 0x08000c
TDS_TYPE_COMPLEX128 = 0x10000d
TDS_TYPE_FLOAT32 = 9
TDS_TYPE_FLOAT64 = 10


_scenarios = []


def scenario(func):
    def as_param():
        result = func()
        return pytest.param(*result, id=func.__name__)
    _scenarios.append(as_param)
    return as_param


def get_scenarios():
    return [f() for f in _scenarios]


@scenario
def single_segment_with_one_channel():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def single_segment_with_two_channels():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def single_segment_with_two_channels_interleaved():
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 3], dtype=np.int32),
        ('group', 'channel2'): np.array([2, 4], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def no_metadata_segment():
    """ Add a segment with two channels, then a second
        segment with the same metadata as before,
        so there is only the lead in and binary data
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocRawData", ),
        "",
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def object_with_no_metadata_in_segment():
    """ Re-use an object without setting any new metadata and
        re-using the data structure
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata_with_repeated_structure("/'group'/'channel1'"),
            channel_metadata_with_repeated_structure("/'group'/'channel2'"),
        ),
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def add_new_channel():
    """ Add a new voltage channel, with the other two channels
        remaining unchanged, so only the new channel is in metadata section
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel3'", TDS_TYPE_INT32, 2),
        ),
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
        "09 00 00 00" "0A 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8], dtype=np.int32),
        ('group', 'channel3'): np.array([9, 10], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def repeated_objects_without_data_in_segment_and_added_object():
    """ Repeated objects with no data in new segment as well as a new channel with data
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata_with_no_data("/'group'/'channel1'"),
            channel_metadata_with_no_data("/'group'/'channel2'"),
            channel_metadata("/'group'/'channel3'", TDS_TYPE_INT32, 2),
        ),
        "05 00 00 00" "06 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4], dtype=np.int32),
        ('group', 'channel3'): np.array([5, 6], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def increase_channel_size():
    """ In the second segment, increase the channel size of one channel
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 4),
        ),
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
        "09 00 00 00" "0A 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8, 9, 10], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def remove_a_channel():
    """ In the second segment, remove a channel.
        We need to write a new object list in this case
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
        ),
        "05 00 00 00" "06 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def alternating_data_objects_with_new_obj_list():
    """ Alternating segments with different objects with data
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata_with_no_data("/'group'/'channel2'"),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata_with_no_data("/'group'/'channel1'"),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata_with_repeated_structure("/'group'/'channel1'"),
            channel_metadata_with_no_data("/'group'/'channel2'"),
        ),
        "05 00 00 00" "06 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata_with_no_data("/'group'/'channel1'"),
            channel_metadata_with_repeated_structure("/'group'/'channel2'"),
        ),
        "07 00 00 00" "08 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def alternating_data_objects_reusing_obj_list():
    """ Alternating segments with different objects with data,
        reusing the object list from the last segment.
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata_with_no_data("/'group'/'channel2'"),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata_with_no_data("/'group'/'channel1'"),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata_with_repeated_structure("/'group'/'channel1'"),
            channel_metadata_with_no_data("/'group'/'channel2'"),
        ),
        "05 00 00 00" "06 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata_with_no_data("/'group'/'channel1'"),
            channel_metadata_with_repeated_structure("/'group'/'channel2'"),
        ),
        "07 00 00 00" "08 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def chunked_segment():
    """ Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
    )
    test_file.add_segment(
        ("kTocRawData", ),
        "",
        "07 00 00 00" "08 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "01 00 00 00" "02 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6, 7, 8, 3, 4], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8, 5, 6, 1, 2], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def chunked_interleaved_segment():
    """ Add interleaved segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
    )
    test_file.add_segment(
        ("kTocRawData", "kTocInterleavedData"),
        "",
        "07 00 00 00" "08 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "01 00 00 00" "02 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 3, 5, 7, 7, 5, 3, 1], dtype=np.int32),
        ('group', 'channel2'): np.array([2, 4, 6, 8, 8, 6, 4, 2], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def less_data_than_expected():
    """ Add segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks,
        but the extra chunk does not have as much data as expected.
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 6], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def less_data_than_expected_interleaved():
    """ Add interleaved data segment and then a repeated segment without
        any lead in or metadata, so data is read in chunks,
        but the extra chunk does not have as much data as expected.
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 3),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 3),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
        "09 00 00 00" "0A 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 3, 5, 7, 9], dtype=np.int32),
        ('group', 'channel2'): np.array([2, 4, 6, 8, 10], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def segment_with_zero_data_length():
    """ kTocRawData is set but data length is zero
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 0),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 0),
        ),
        ""
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def incomplete_last_segment():
    """ Test incomplete last segment, eg. if LabView crashed
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
    )
    test_file.add_segment(
        ("kTocRawData", ),
        "",
        "09 00 00 00" "0A 00 00 00"
        "0B 00 00 00" "0C 00 00 00",
        incomplete=True
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 2, 5, 6, 9, 10], dtype=np.int32),
        ('group', 'channel2'): np.array([3, 4, 7, 8, 11, 12], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def incomplete_last_row_of_interleaved_data():
    """ Test incomplete last row of interleaved data
    """
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 2),
        ),
        "01 00 00 00" "02 00 00 00"
        "03 00 00 00" "04 00 00 00"
        "05 00 00 00" "06 00 00 00"
        "07 00 00 00" "08 00 00 00"
        "09 00 00 00" "0A 00 00 00"
        "0B 00 00 00",
        incomplete=True
    )
    expected_data = {
        ('group', 'channel1'): np.array([1, 3, 5, 7, 9], dtype=np.int32),
        ('group', 'channel2'): np.array([2, 4, 6, 8, 10], dtype=np.int32),
    }
    return test_file, expected_data


@scenario
def float_data():
    """ Test reading a file with float valued data
    """
    single_arr = np.array([0.123, 0.234, 0.345, 0.456], dtype=np.float32)
    double_arr = np.array([0.987, 0.876, 0.765, 0.654], dtype=np.double)
    data = ""
    for num in single_arr[0:2]:
        data += hexlify_value("<f", num)
    for num in double_arr[0:2]:
        data += hexlify_value("<d", num)
    for num in single_arr[2:4]:
        data += hexlify_value("<f", num)
    for num in double_arr[2:4]:
        data += hexlify_value("<d", num)

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'single_channel'", TDS_TYPE_FLOAT32, 2),
            channel_metadata("/'group'/'double_channel'", TDS_TYPE_FLOAT64, 2),
        ),
        data
    )
    expected_data = {
        ('group', 'single_channel'): single_arr,
        ('group', 'double_channel'): double_arr,
    }
    return test_file, expected_data


@scenario
def complex_data():
    """ Test reading a file with complex valued data
    """
    complex_single_arr = np.array([1+2j, 3+4j], dtype=np.complex64)
    complex_double_arr = np.array([5+6j, 7+8j], dtype=np.complex128)
    data = ""
    for num in complex_single_arr:
        data += hexlify_value("<f", num.real)
        data += hexlify_value("<f", num.imag)
    for num in complex_double_arr:
        data += hexlify_value("<d", num.real)
        data += hexlify_value("<d", num.imag)

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'complex_single_channel'", TDS_TYPE_COMPLEX64, 2),
            channel_metadata("/'group'/'complex_double_channel'", TDS_TYPE_COMPLEX128, 2),
        ),
        data
    )
    expected_data = {
        ('group', 'complex_single_channel'): complex_single_arr,
        ('group', 'complex_double_channel'): complex_double_arr,
    }
    return test_file, expected_data


@scenario
def scaled_data():
    properties = {
        "NI_Number_Of_Scales":
            (3, "01 00 00 00"),
        "NI_Scale[0]_Scale_Type":
            (0x20, hexlify_value("<I", len("Linear")) + string_hexlify("Linear")),
        "NI_Scale[0]_Linear_Slope":
            (10, hexlify_value("<d", 2.0)),
        "NI_Scale[0]_Linear_Y_Intercept":
            (10, hexlify_value("<d", 10.0))
    }
    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 2, properties),
        ),
        "01 00 00 00" "02 00 00 00"
    )
    expected_data = {
        ('group', 'channel1'): np.array([12, 14], dtype=np.float64),
    }
    return test_file, expected_data


@scenario
def timestamp_data():
    """Test reading contiguous timestamp data
    """

    times = [
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]

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

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    test_file.add_segment(toc, metadata, _timestamp_data(times))

    expected_data = {
        ('Group', 'TimeChannel1'): np.array([times[0], times[1]]),
        ('Group', 'TimeChannel2'): np.array([times[2], times[3]]),
    }

    return test_file, expected_data


@scenario
def interleaved_timestamp_data():
    """Test reading interleaved timestamp data
    """

    times = [
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]

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

    test_file = GeneratedFile()
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData")
    test_file.add_segment(toc, metadata, _timestamp_data(times))

    expected_data = {
        ('Group', 'TimeChannel1'): np.array([times[0], times[2]]),
        ('Group', 'TimeChannel2'): np.array([times[1], times[3]]),
    }

    return test_file, expected_data


def _timestamp_data(times):
    epoch = np.datetime64('1904-01-01T00:00:00')

    def total_seconds(td):
        return int(td / np.timedelta64(1, 's'))

    def microseconds(dt):
        diff = dt - epoch
        secs = total_seconds(diff)
        remainder = diff - np.timedelta64(secs, 's')
        return int(remainder / np.timedelta64(1, 'us'))

    seconds = [total_seconds(t - epoch) for t in times]
    fractions = [
        int(float(microseconds(t)) * 2 ** 58 / 5 ** 6)
        for t in times]

    data = ""
    for f, s in zip(fractions, seconds):
        data += hexlify_value("<Q", f)
        data += hexlify_value("<q", s)
    return data
