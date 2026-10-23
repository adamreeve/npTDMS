import os
from io import BytesIO
import numpy as np
import pytest

from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject, extract_segments
from nptdms.test.util import (
    GeneratedFile,
    hexlify_value,
    string_hexlify,
    segment_objects_metadata,
    channel_metadata,
    channel_metadata_with_no_data,
    channel_metadata_with_repeated_structure)
from nptdms.test.scenarios import TDS_TYPE_INT32, timestamp_data_chunk


@pytest.mark.benchmark(group='read-all-data')
def test_read_contiguous_data(benchmark):
    """ Benchmark reading a file with multiple channels of contiguous data
    """
    tdms_file = benchmark(read_from_start, get_contiguous_file().get_bytes_io_file())

    np.testing.assert_equal(tdms_file['group']['channel1'][:], np.repeat([1], 10000))
    np.testing.assert_equal(tdms_file['group']['channel2'][:], np.repeat([2], 10000))
    np.testing.assert_equal(tdms_file['group']['channel3'][:], np.repeat([3], 10000))
    np.testing.assert_equal(tdms_file['group']['channel4'][:], np.repeat([4], 10000))


@pytest.mark.benchmark(group='read-all-data')
def test_read_interleaved_data(benchmark):
    """ Benchmark reading a file with interleaved data
    """
    tdms_file = benchmark(read_from_start, get_interleaved_file().get_bytes_io_file())

    np.testing.assert_equal(tdms_file['group']['channel1'][:], np.repeat([1], 10000))
    np.testing.assert_equal(tdms_file['group']['channel2'][:], np.repeat([2], 10000))
    np.testing.assert_equal(tdms_file['group']['channel3'][:], np.repeat([3], 10000))
    np.testing.assert_equal(tdms_file['group']['channel4'][:], np.repeat([4], 10000))


@pytest.mark.benchmark(group='read-all-channel')
def test_read_contiguous_data_channel(benchmark):
    """ Benchmark reading a single channel's data from a contiguous data file
    """
    with TdmsFile.open(get_contiguous_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(read_channel_data, channel)

        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-all-channel')
def test_read_interleaved_data_channel(benchmark):
    """ Benchmark reading a single channel's data from an interleaved data file
    """
    with TdmsFile.open(get_interleaved_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(read_channel_data, channel)

        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-all-channel')
def test_stream_contiguous_data_channel(benchmark):
    """ Benchmark streaming channel data from a contiguous data file
    """
    with TdmsFile.open(get_contiguous_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(stream_chunks, channel)

        channel_data = np.concatenate(channel_data)
        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-all-channel')
def test_stream_interleaved_data_channel(benchmark):
    """ Benchmark streaming channel data from an interleaved data file
    """
    with TdmsFile.open(get_interleaved_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(stream_chunks, channel)

        channel_data = np.concatenate(channel_data)
        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='slice-channel')
def test_slice_contiguous_data_channel(benchmark):
    """ Benchmark reading a slice of data from a contiguous data file
    """
    with TdmsFile.open(get_contiguous_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(get_slice, channel, 5555, 6555)

        expected_data = np.repeat([3], 1000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='slice-channel')
def test_slice_interleaved_data_channel(benchmark):
    """ Benchmark reading a slice of data from an interleaved data file
    """
    with TdmsFile.open(get_interleaved_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = benchmark(get_slice, channel, 5555, 6555)

        expected_data = np.repeat([3], 1000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-all-channel')
def test_index_contiguous_data_channel(benchmark):
    """ Benchmark reading a data from a contiguous data file using integer indices
    """
    with TdmsFile.open(get_contiguous_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = np.zeros(10000, dtype=channel.dtype)
        benchmark(index_values, channel, channel_data)

        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-all-channel')
def test_index_interleaved_data_channel(benchmark):
    """ Benchmark reading a data from a interleaved data file using integer indices
    """
    with TdmsFile.open(get_interleaved_file().get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel3']
        channel_data = np.zeros(10000, dtype=channel.dtype)
        benchmark(index_values, channel, channel_data)

        expected_data = np.repeat([3], 10000)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-scaled-channel')
def test_stream_scaled_data_chunks(benchmark):
    """ Benchmark streaming channel data when the data is scaled
    """
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
    data_array = np.arange(0, 1000, dtype=np.dtype('int32'))
    data = data_array.tobytes()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 100, properties),
        ),
        data, binary_data=True
    )
    for _ in range(0, 9):
        test_file.add_segment(
            ("kTocRawData", ), "", data, binary_data=True)

    with TdmsFile.open(test_file.get_bytes_io_file()) as tdms_file:
        channel = tdms_file['group']['channel1']
        channel_data = benchmark(stream_chunks, channel)

        channel_data = np.concatenate(channel_data)
        expected_data = np.tile(10.0 + 2.0 * data_array, 10)
        np.testing.assert_equal(channel_data, expected_data)


@pytest.mark.benchmark(group='read-timestamp-data')
def test_read_timestamp_data(benchmark):
    """ Benchmark reading a file with timestamp data
    """
    timestamps = np.tile(np.array([
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]), 200)
    data = timestamp_data_chunk(timestamps)

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 0x44, 200, {}),
        ),
        data
    )

    tdms_file = benchmark(read_from_start, test_file.get_bytes_io_file())

    np.testing.assert_equal(tdms_file['group']['channel1'][:], timestamps)


@pytest.mark.benchmark(group='read-timestamp-data')
def test_read_interleaved_timestamp_data(benchmark):
    """ Benchmark reading a file with interleaved timestamp data
    """
    timestamps = np.tile(np.array([
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
        np.datetime64('2012-08-23T00:00:00.123', 'us'),
        np.datetime64('2012-08-23T01:02:03.456', 'us'),
        np.datetime64('2012-08-23T12:00:00.0', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
        np.datetime64('2012-08-23T12:02:03.9999', 'us'),
    ]), 100)
    data = timestamp_data_chunk(timestamps)

    test_file = GeneratedFile()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", 0x44, 100, {}),
            channel_metadata("/'group'/'channel2'", 0x44, 100, {}),
        ),
        data
    )

    tdms_file = benchmark(read_from_start, test_file.get_bytes_io_file())

    np.testing.assert_equal(tdms_file['group']['channel1'][:], timestamps[0::2])
    np.testing.assert_equal(tdms_file['group']['channel2'][:], timestamps[1::2])


@pytest.mark.benchmark(group='read-metadata')
def test_complex_metadata_reading(benchmark):
    """ Benchmark reading metadata for a file with many channels and segments with alternating sets of objects
    """
    test_file = GeneratedFile()
    data = np.array([0] * 5, dtype=np.dtype('int32')).tobytes()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel0'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel3'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel4'", TDS_TYPE_INT32, 1),
        ),
        data, binary_data=True
    )
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel5'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel6'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel7'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel8'", TDS_TYPE_INT32, 1),
            channel_metadata("/'group'/'channel9'", TDS_TYPE_INT32, 1),
        ),
        data, binary_data=True
    )
    for _ in range(9):
        test_file.add_segment(
            ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
            segment_objects_metadata(
                channel_metadata_with_no_data("/'group'/'channel0'"),
                channel_metadata_with_no_data("/'group'/'channel1'"),
                channel_metadata_with_no_data("/'group'/'channel2'"),
                channel_metadata_with_no_data("/'group'/'channel3'"),
                channel_metadata_with_no_data("/'group'/'channel4'"),
                channel_metadata_with_repeated_structure("/'group'/'channel5'"),
                channel_metadata_with_repeated_structure("/'group'/'channel6'"),
                channel_metadata_with_repeated_structure("/'group'/'channel7'"),
                channel_metadata_with_repeated_structure("/'group'/'channel8'"),
                channel_metadata_with_repeated_structure("/'group'/'channel9'"),
            ),
            data, binary_data=True
        )
        test_file.add_segment(
            ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
            segment_objects_metadata(
                channel_metadata_with_repeated_structure("/'group'/'channel0'"),
                channel_metadata_with_repeated_structure("/'group'/'channel1'"),
                channel_metadata_with_repeated_structure("/'group'/'channel2'"),
                channel_metadata_with_repeated_structure("/'group'/'channel3'"),
                channel_metadata_with_repeated_structure("/'group'/'channel4'"),
                channel_metadata_with_no_data("/'group'/'channel5'"),
                channel_metadata_with_no_data("/'group'/'channel6'"),
                channel_metadata_with_no_data("/'group'/'channel7'"),
                channel_metadata_with_no_data("/'group'/'channel8'"),
                channel_metadata_with_no_data("/'group'/'channel9'"),
            ),
            data, binary_data=True
        )

    tdms_file = benchmark(read_metadata_from_start, test_file.get_bytes_io_file())

    assert len(tdms_file) == 1
    assert len(tdms_file['group']) == 10
    for channel_num in range(10):
        assert len(tdms_file['group']['channel{0}'.format(channel_num)]) == 10


@pytest.mark.benchmark(group='read-metadata')
def test_read_metadata_with_one_channel_per_segment(benchmark):
    """ Benchmark reading metadata for a file where every segment contains the metadata and data of one channel
    """
    tdms_file = benchmark(read_metadata_from_start, get_one_channel_per_segment_file())

    assert len(tdms_file['group']['channel7']) == 64 * 150


@pytest.mark.benchmark(group='read-metadata')
def test_read_metadata_of_many_channels_without_metadata_in_segments(benchmark):
    """ Benchmark reading metadata for a file with many channels and many segments without metadata
    """
    tdms_file = benchmark(read_metadata_from_start, get_many_segments_without_metadata_file())

    assert len(tdms_file['group']['channel7']) == 10 * 5000


@pytest.mark.benchmark(group='read-channel-from-one-channel-per-segment')
def test_read_channel_from_one_channel_per_segment(benchmark):
    """ Benchmark reading a channel from a file where every segment contains one channel
    """
    channel_data = benchmark(read_channel_from_start, get_one_channel_per_segment_file(), 'group', 'channel7')

    np.testing.assert_equal(channel_data, np.tile(np.arange(64, dtype=np.int32), 150))


@pytest.mark.benchmark(group='read-channel-from-one-channel-per-segment')
def test_read_channel_from_byte_ranges(benchmark):
    """ Benchmark reading a channel from only its byte ranges, from a file where every segment contains one channel
    """
    file = get_one_channel_per_segment_file()
    file.seek(0, os.SEEK_SET)
    byte_ranges = TdmsFile.read_metadata(file, byte_ranges=True)['group']['channel7'].byte_ranges()

    channel_data = benchmark(read_channel_from_byte_ranges, file, byte_ranges, 'group', 'channel7')

    np.testing.assert_equal(channel_data, np.tile(np.arange(64, dtype=np.int32), 150))


@pytest.mark.benchmark(group='write-many-channels')
def test_write_many_channels_low_data_per_channel(benchmark):
    """Benchmark writing many channels with a small amount of data each."""
    num_channels = 128
    samples_per_channel = 8

    output_file = benchmark(
        write_segment,
        make_write_segment(num_channels, samples_per_channel),
    )

    output_file.seek(0)

    with TdmsFile.open(output_file) as tdms_file:
        group = tdms_file['group']
        expected = np.arange(samples_per_channel, dtype=np.int32)

        assert len(group) == num_channels
        np.testing.assert_array_equal(group['channel0'][:], expected)
        np.testing.assert_array_equal(
            group['channel{0}'.format(num_channels - 1)][:],
            expected + (num_channels - 1),
        )


@pytest.mark.benchmark(group='write-few-channels')
def test_write_few_channels_data_per_channel(benchmark):
    """Benchmark writing a few channels with a large amount of data each."""
    num_channels = 4
    samples_per_channel = 8192

    output_file = benchmark(
        write_segment,
        make_write_segment(num_channels, samples_per_channel),
    )

    output_file.seek(0)

    with TdmsFile.open(output_file) as tdms_file:
        group = tdms_file['group']
        expected = np.arange(samples_per_channel, dtype=np.int32)

        assert len(group) == num_channels
        np.testing.assert_array_equal(group['channel0'][:], expected)
        np.testing.assert_array_equal(
            group['channel{0}'.format(num_channels - 1)][:],
            expected + (num_channels - 1),
        )


@pytest.mark.benchmark(group='write-single-channel')
def test_write_single_channel_high_data(benchmark):
    """Benchmark writing a single channels with a high amount of data."""
    num_channels = 1
    samples_per_channel = 1048576

    output_file = benchmark(
        write_segment,
        make_write_segment(num_channels, samples_per_channel),
    )

    output_file.seek(0)

    with TdmsFile.open(output_file) as tdms_file:
        group = tdms_file['group']
        expected = np.arange(samples_per_channel, dtype=np.int32)

        assert len(group) == num_channels
        np.testing.assert_array_equal(group['channel0'][:], expected)
        np.testing.assert_array_equal(
            group['channel{0}'.format(num_channels - 1)][:],
            expected + (num_channels - 1),
        )


def get_contiguous_file():
    test_file = GeneratedFile()
    data_chunk = np.repeat(np.array([1, 2, 3, 4], dtype=np.dtype('int32')), 100)
    data_array = np.tile(data_chunk, 10)
    data = data_array.tobytes()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel3'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel4'", TDS_TYPE_INT32, 100),
        ),
        data, binary_data=True
    )
    for _ in range(0, 9):
        test_file.add_segment(
            ("kTocRawData", ), "", data, binary_data=True)
    return test_file


def get_interleaved_file():
    test_file = GeneratedFile()
    data_array = np.tile(np.array([1, 2, 3, 4], dtype=np.dtype('int32')), 1000)
    data = data_array.tobytes()
    test_file.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList", "kTocInterleavedData"),
        segment_objects_metadata(
            channel_metadata("/'group'/'channel1'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel2'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel3'", TDS_TYPE_INT32, 100),
            channel_metadata("/'group'/'channel4'", TDS_TYPE_INT32, 100),
        ),
        data, binary_data=True
    )
    for _ in range(0, 9):
        test_file.add_segment(
            ("kTocRawData", "kTocInterleavedData"), "", data, binary_data=True)
    return test_file


def get_one_channel_per_segment_file():
    file = BytesIO()
    with TdmsWriter(file) as writer:
        for i in range(3000):
            writer.write_segment([ChannelObject(
                'group', 'channel%d' % (i % 20), np.arange(64, dtype=np.int32), properties={'index': i})])
    return file


def get_many_segments_without_metadata_file():
    data = np.zeros(10 * 50, dtype=np.int32).tobytes()
    first_segment = GeneratedFile()
    first_segment.add_segment(
        ("kTocMetaData", "kTocRawData", "kTocNewObjList"),
        segment_objects_metadata(*[
            channel_metadata("/'group'/'channel%d'" % i, TDS_TYPE_INT32, 10) for i in range(50)]),
        data, binary_data=True)
    segment = GeneratedFile()
    segment.add_segment(("kTocRawData", ), "", data, binary_data=True)
    return BytesIO(first_segment.get_bytes_io_file().getvalue() + segment.get_bytes_io_file().getvalue() * 4999)


def read_channel_from_start(file, group, channel):
    file.seek(0, os.SEEK_SET)
    with TdmsFile.open(file) as tdms_file:
        return tdms_file[group][channel][:]


def read_channel_from_byte_ranges(file, byte_ranges, group, channel):
    return TdmsFile.read(BytesIO(extract_segments(file, byte_ranges)))[group][channel][:]


def read_from_start(file):
    file.seek(0, os.SEEK_SET)
    return TdmsFile.read(file)


def read_metadata_from_start(file):
    file.seek(0, os.SEEK_SET)
    return TdmsFile.read_metadata(file)


def read_channel_data(chan):
    return chan[:]


def stream_chunks(chan):
    all_data = []
    for chunk in chan.data_chunks():
        all_data.append(chunk[:])
    return all_data


def get_slice(chan, start, stop):
    return chan[start:stop]


def index_values(chan, target):
    for i in range(len(chan)):
        target[i] = chan[i]


def make_write_segment(num_channels, samples_per_channel):
    segment = [RootObject(), GroupObject("group")]
    base_data = np.arange(samples_per_channel, dtype=np.int32)

    for channel_index in range(num_channels):
        segment.append(
            ChannelObject(
                "group",
                "channel{0}".format(channel_index),
                base_data + channel_index,
            )
        )

    return segment


def write_segment(segment):
    output_file = BytesIO()

    with TdmsWriter(output_file) as tdms_writer:
        tdms_writer.write_segment(segment)

    return output_file
