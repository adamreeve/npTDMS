"""Test writing TDMS files"""

from datetime import datetime
from io import BytesIO
import numpy as np
import os
import tempfile

from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject, types


def test_can_read_tdms_file_after_writing():
    a_input = np.linspace(0.0, 1.0, 100, dtype='float32')
    b_input = np.linspace(0.0, 100.0, 100, dtype='float64')

    a_segment = ChannelObject("group", "a", a_input)
    b_segment = ChannelObject("group", "b", b_input)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([a_segment, b_segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    a_output = tdms_file["group"]["a"]
    b_output = tdms_file["group"]["b"]

    assert a_output.data_type == types.SingleFloat
    assert b_output.data_type == types.DoubleFloat
    assert a_output.dtype == np.dtype('float32')
    assert b_output.dtype == np.dtype('float64')
    assert len(a_output) == len(a_input)
    assert len(b_output) == len(b_input)
    assert (a_output[:] == a_input).all()
    assert (b_output[:] == b_input).all()


def test_can_read_tdms_file_properties_after_writing():
    test_time = np.datetime64('2019-11-19T15:30:00')

    a_segment = RootObject(properties={
        "prop1": "foo",
        "prop2": 3,
    })
    b_segment = GroupObject("group_name", properties={
        "prop3": 1.2345,
        "prop4": test_time,
    })

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([a_segment, b_segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    file_properties = tdms_file.properties
    b_output = tdms_file["group_name"]

    assert "prop1" in file_properties, "prop1 not found"
    assert "prop2" in file_properties, "prop2 not found"
    assert "prop3" in b_output.properties, "prop3 not found"
    assert "prop4" in b_output.properties, "prop4 not found"
    assert file_properties["prop1"] == "foo"
    assert file_properties["prop2"] == 3
    assert b_output.properties["prop3"] == 1.2345
    assert b_output.properties["prop4"] == test_time


def test_can_write_multiple_segments():
    input_1 = np.linspace(0.0, 1.0, 10)
    input_2 = np.linspace(2.0, 3.0, 10)

    segment_1 = ChannelObject("group", "a", input_1)
    segment_2 = ChannelObject("group", "a", input_2)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment_1])
        tdms_writer.write_segment([segment_2])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["a"].data

    expected_data = np.append(input_1, input_2)
    assert len(output_data) == len(expected_data)
    assert (output_data == expected_data).all()


def test_can_write_to_file_using_path():
    input_1 = np.linspace(0.0, 1.0, 10)
    segment = ChannelObject("group", "a", input_1)

    tempdir = tempfile.mkdtemp()
    temppath = "%s/test_file.tdms" % tempdir
    try:
        with TdmsWriter(temppath) as tdms_writer:
            tdms_writer.write_segment([segment])
    finally:
        if os.path.exists(temppath):
            os.remove(temppath)
        os.rmdir(tempdir)


def test_can_append_to_file_using_path():
    input_1 = np.linspace(0.0, 1.0, 10)
    input_2 = np.linspace(1.0, 2.0, 10)
    segment_1 = ChannelObject("group", "a", input_1)
    segment_2 = ChannelObject("group", "a", input_2)

    tempdir = tempfile.mkdtemp()
    temppath = "%s/test_file.tdms" % tempdir
    try:
        with TdmsWriter(temppath) as tdms_writer:
            tdms_writer.write_segment([segment_1])
        with TdmsWriter(temppath, 'a') as tdms_writer:
            tdms_writer.write_segment([segment_2])

        tdms_file = TdmsFile(temppath)

        output = tdms_file["group"]["a"].data

        assert len(output) == 20
        np.testing.assert_almost_equal(
            output, np.concatenate([input_1, input_2]))

    finally:
        if os.path.exists(temppath):
            os.remove(temppath)
        os.rmdir(tempdir)


def test_can_write_to_file_using_open_file():
    input_1 = np.linspace(0.0, 1.0, 10)
    segment = ChannelObject("group", "a", input_1)

    with tempfile.NamedTemporaryFile(delete=True) as output_file:
        with TdmsWriter(output_file.file) as tdms_writer:
            tdms_writer.write_segment([segment])


def test_can_write_tdms_objects_read_from_file():
    group_segment = GroupObject("group", properties={
        "prop1": "bar"
    })
    input_data = np.linspace(0.0, 1.0, 10)
    channel_segment = ChannelObject("group", "a", input_data, properties={
        "prop1": "foo",
        "prop2": 3,
    })

    tempdir = tempfile.mkdtemp()
    temppath = "%s/test_file.tdms" % tempdir
    try:
        with TdmsWriter(temppath) as tdms_writer:
            tdms_writer.write_segment([group_segment, channel_segment])

        tdms_file = TdmsFile(temppath)
        read_group = tdms_file["group"]
        read_channel = tdms_file["group"]["a"]

        with TdmsWriter(temppath) as tdms_writer:
            tdms_writer.write_segment([read_group, read_channel])

        tdms_file = TdmsFile(temppath)
        read_group = tdms_file["group"]
        read_channel = tdms_file["group"]["a"]

        assert read_group.properties["prop1"] == "bar"

        assert len(read_channel.data) == 10
        np.testing.assert_almost_equal(read_channel.data, input_data)
        assert read_channel.properties["prop1"] == "foo"
        assert read_channel.properties["prop2"] == 3

    finally:
        if os.path.exists(temppath):
            os.remove(temppath)
        os.rmdir(tempdir)


def test_can_write_timestamp_data():
    input_data = [
        np.datetime64('2017-07-09T12:35:00.00'),
        np.datetime64('2017-07-09T12:36:00.00'),
        np.datetime64('2017-07-09T12:37:00.00'),
        ]

    segment = ChannelObject("group", "timedata", input_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["timedata"].data

    assert len(output_data) == 3
    assert output_data[0] == input_data[0]
    assert output_data[1] == input_data[1]
    assert output_data[2] == input_data[2]


def test_can_write_timestamp_data_with_datetimes():
    input_data = [
        datetime(2017, 7, 9, 12, 35, 0),
        datetime(2017, 7, 9, 12, 36, 0),
        datetime(2017, 7, 9, 12, 37, 0)]
    expected_data = np.array([
        '2017-07-09T12:35:00',
        '2017-07-09T12:36:00',
        '2017-07-09T12:37:00'], dtype='datetime64')

    segment = ChannelObject("group", "timedata", input_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["timedata"].data

    assert len(output_data) == 3
    assert output_data[0] == expected_data[0]
    assert output_data[1] == expected_data[1]
    assert output_data[2] == expected_data[2]


def test_can_write_numpy_timestamp_data_with_dates():
    input_data = np.array([
        '2017-07-09',
        '2017-07-09',
        '2017-07-09'], dtype='datetime64')

    segment = ChannelObject("group", "timedata", input_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["timedata"].data

    assert len(output_data) == 3
    assert output_data[0] == input_data[0]
    assert output_data[1] == input_data[1]
    assert output_data[2] == input_data[2]


def test_can_write_string_data():
    input_data = [
        "hello world",
        u"\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"]

    segment = ChannelObject("group", "string_data", input_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["string_data"].data

    assert len(output_data) == 2
    assert output_data[0] == input_data[0]
    assert output_data[1] == input_data[1]


def test_can_write_floats_from_list():
    input_data = [1.0, 2.0, 3.0]

    segment = ChannelObject("group", "data", input_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["data"].data

    assert output_data.dtype == np.float64
    assert len(output_data) == 3
    assert output_data[0] == input_data[0]
    assert output_data[1] == input_data[1]
    assert output_data[2] == input_data[2]


def test_can_write_ints_from_list():
    test_cases = [
        (np.int8, types.Int8, [0, 1]),
        (np.int8, types.Int8, [-2 ** 7, 0]),
        (np.int8, types.Int8, [0, 2 ** 7 - 1]),

        (np.uint8, types.Uint8, [0, 2 ** 7]),
        (np.uint8, types.Uint8, [0, 2 ** 8 - 1]),

        (np.int16, types.Int16, [-2 ** 15, 0]),
        (np.int16, types.Int16, [0, 2 ** 15 - 1]),

        (np.uint16, types.Uint16, [0, 2 ** 15]),
        (np.uint16, types.Uint16, [0, 2 ** 16 - 1]),

        (np.int32, types.Int32, [-2 ** 31, 0]),
        (np.int32, types.Int32, [0, 2 ** 31 - 1]),

        (np.uint32, types.Uint32, [0, 2 ** 31]),
        (np.uint32, types.Uint32, [0, 2 ** 32 - 1]),

        (np.int64, types.Int64, [-2 ** 63, 0]),
        (np.int64, types.Int64, [0, 2 ** 63 - 1]),

        (np.uint64, types.Uint64, [0, 2 ** 63]),
        (np.uint64, types.Uint64, [0, 2 ** 64 - 1]),
    ]

    for expected_dtype, expected_tdms_type, input_data in test_cases:
        test_case = "data = %s, expected_dtype = %s" % (
            input_data, expected_dtype)
        segment = ChannelObject("group", "data", input_data)

        output_file = BytesIO()
        with TdmsWriter(output_file) as tdms_writer:
            tdms_writer.write_segment([segment])

        output_file.seek(0)
        tdms_file = TdmsFile(output_file)

        output_data = tdms_file["group"]["data"]

        assert output_data.data_type == expected_tdms_type, test_case
        assert output_data.dtype == expected_dtype, test_case
        assert len(output_data) == len(input_data), test_case
        for (input_val, output_val) in zip(input_data, output_data):
            assert output_val == input_val, test_case


def test_can_write_complex():
    input_complex64_data = np.array([1+2j, 3+4j], np.complex64)
    input_complex128_data = np.array([5+6j, 7+8j], np.complex128)

    complex64_segment = ChannelObject(
            "group", "complex64_data", input_complex64_data)
    complex128_segment = ChannelObject(
            "group", "complex128_data", input_complex128_data)

    output_file = BytesIO()
    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment([complex64_segment])
        tdms_writer.write_segment([complex128_segment])

    output_file.seek(0)
    tdms_file = TdmsFile(output_file)

    output_data = tdms_file["group"]["complex64_data"]
    assert output_data.data_type == types.ComplexSingleFloat
    assert output_data.dtype == np.complex64
    assert len(output_data) == 2
    assert output_data[0] == input_complex64_data[0]
    assert output_data[1] == input_complex64_data[1]

    output_data = tdms_file["group"]["complex128_data"]
    assert output_data.data_type == types.ComplexDoubleFloat
    assert output_data.dtype == np.complex128
    assert len(output_data) == 2
    assert output_data[0] == input_complex128_data[0]
    assert output_data[1] == input_complex128_data[1]
