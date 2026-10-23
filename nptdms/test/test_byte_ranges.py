""" Tests for reading objects from only the byte ranges of a TDMS file they are stored in
"""

from io import BytesIO
import os
import sqlite3

from hypothesis import assume, given, settings, strategies
import numpy as np
import pytest

from nptdms import TdmsFile, TdmsGroup, TdmsChannel, TdmsWriter, ChannelObject
from nptdms import merge_byte_ranges, extract_segments
from nptdms.reader import _insert_byte_range
from nptdms.test import scenarios
from nptdms.test.scenarios import TDS_TYPE_INT32
from nptdms.test.util import (
    GeneratedFile,
    compare_arrays,
    channel_metadata,
    channel_metadata_with_no_data,
    channel_metadata_with_repeated_structure,
    hexlify_value,
    segment_objects_metadata,
)


DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/data'
NEW_LIST = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
INCREMENTAL = ("kTocMetaData", "kTocRawData")
NO_METADATA = ("kTocRawData", )
CHANNELS = ["/'group'/'a'", "/'group'/'b'", "/'group'/'c'"]


def assert_objects_read_from_byte_ranges(data):
    """ Check that every object in a file can be read from only its byte ranges
    """
    tdms_file = TdmsFile.read(BytesIO(data), byte_ranges=True)
    for obj in [tdms_file] + [o for group in tdms_file.groups() for o in [group] + group.channels()]:
        byte_ranges = obj.byte_ranges()
        if not byte_ranges:
            # The object is not in the file's metadata
            assert len(obj.properties) == 0
            continue
        part = TdmsFile.read(BytesIO(extract_segments(BytesIO(data), byte_ranges)))
        if isinstance(obj, TdmsChannel):
            part_obj = part[obj.group_name][obj.name]
        else:
            part_obj = part[obj.name] if isinstance(obj, TdmsGroup) else part
        assert list(part_obj.properties.items()) == list(obj.properties.items())
        if isinstance(obj, TdmsChannel):
            assert part_obj.data_type == obj.data_type
            expected = obj.read_data(scaled=False)
            actual = part_obj.read_data(scaled=False)
            if isinstance(expected, dict):
                # DAQmx raw data
                for scaler_id in expected:
                    np.testing.assert_equal(actual[scaler_id], expected[scaler_id])
            else:
                assert actual.dtype == expected.dtype
                compare_arrays(actual, expected)


def _segment_indexes(tdms_file, byte_ranges):
    """ Get the indexes of the segments in byte ranges, for readable test expectations
    """
    return [
        i for (i, segment) in enumerate(tdms_file._reader._segments)
        if any(offset <= segment.position < offset + length for (offset, length) in byte_ranges)]


@pytest.mark.parametrize("test_file,expected_data", scenarios.get_scenarios())
def test_read_objects_from_byte_ranges(test_file, expected_data):
    assert_objects_read_from_byte_ranges(test_file.get_bytes_io_file().getvalue())


@pytest.mark.parametrize("file_name", ["Digital_Input.tdms", "big_endian.tdms", "raw1.tdms", "raw_timestamps.tdms"])
def test_read_objects_of_example_files_from_byte_ranges(file_name):
    with open(os.path.join(DATA_DIR, file_name), 'rb') as f:
        assert_objects_read_from_byte_ranges(f.read())


def _int32_data(*values):
    return "".join(hexlify_value("<i", v) for v in values)


def _properties(name, value):
    return {name: (TDS_TYPE_INT32, hexlify_value("<i", value))}


_FIRST_SEGMENT = (NEW_LIST, [channel_metadata(CHANNELS[0], TDS_TYPE_INT32, 1),
                             channel_metadata(CHANNELS[1], TDS_TYPE_INT32, 1)], _int32_data(1, 2))


@pytest.mark.parametrize("segments,path,expected_segments", [
    pytest.param(
        [_FIRST_SEGMENT] + [(NO_METADATA, [], _int32_data(3, 4))] * 4 + [
            (INCREMENTAL, [channel_metadata(CHANNELS[2], TDS_TYPE_INT32, 1)], _int32_data(5, 6, 7)),
            (NO_METADATA, [], _int32_data(8, 9, 10))],
        CHANNELS[2], [0, 5, 6], id="channel_added_later"),
    pytest.param([
        _FIRST_SEGMENT,
        (INCREMENTAL, [channel_metadata_with_no_data(CHANNELS[1])], _int32_data(3)),
        (INCREMENTAL, [channel_metadata_with_repeated_structure(CHANNELS[1])], ""),
        (NO_METADATA, [], _int32_data(4, 5)),
    ], CHANNELS[1], [0, 1, 2, 3], id="reactivated_without_data"),
    pytest.param([
        _FIRST_SEGMENT,
        (INCREMENTAL, [channel_metadata_with_no_data(CHANNELS[1])], ""),
        (NO_METADATA, [], _int32_data(4)),
    ], CHANNELS[0], [0, 1, 2], id="deactivated_without_data"),
    pytest.param([
        _FIRST_SEGMENT,
        (INCREMENTAL, [channel_metadata(CHANNELS[1], TDS_TYPE_INT32, 2)], ""),
        (NO_METADATA, [], _int32_data(4, 5, 6)),
    ], CHANNELS[0], [0, 1, 2], id="new_index_without_data"),
    pytest.param([
        _FIRST_SEGMENT,
        (NEW_LIST, [channel_metadata_with_repeated_structure(CHANNELS[0]),
                    channel_metadata_with_no_data(CHANNELS[1])], _int32_data(3)),
        (NEW_LIST, [channel_metadata(CHANNELS[2], TDS_TYPE_INT32, 1)], _int32_data(4)),
        (INCREMENTAL, [channel_metadata_with_repeated_structure(CHANNELS[1])], _int32_data(5, 6)),
    ], CHANNELS[1], [0, 2, 3], id="reused_raw_data_index"),
    pytest.param([
        (NEW_LIST, [channel_metadata_with_no_data("/'group'", _properties('x', 1)),
                    channel_metadata(CHANNELS[0], TDS_TYPE_INT32, 1)], _int32_data(1)),
        (NO_METADATA, [], _int32_data(2)),
        (INCREMENTAL, [channel_metadata_with_no_data("/'group'", _properties('x', 2))], _int32_data(3)),
    ], "/'group'", [0, 2], id="all_property_segments"),
    pytest.param([
        (NEW_LIST, [channel_metadata(CHANNELS[0], TDS_TYPE_INT32, 1)], _int32_data(1)),
        (INCREMENTAL, [channel_metadata_with_no_data("/'group'")], _int32_data(2)),
    ], "/'group'", [0, 1], id="first_appearance"),
    pytest.param([
        (NEW_LIST, [channel_metadata(CHANNELS[0], TDS_TYPE_INT32, 1),
                    channel_metadata_with_no_data(CHANNELS[1])], _int32_data(1)),
        (NEW_LIST, [channel_metadata_with_repeated_structure(CHANNELS[0]),
                    channel_metadata(CHANNELS[1], TDS_TYPE_INT32, 1)], ""),
        (NEW_LIST, [channel_metadata_with_repeated_structure(CHANNELS[0])], _int32_data(2)),
    ], CHANNELS[1], [0, 1], id="data_type_without_data"),
])
def test_byte_ranges_include_dependencies(segments, path, expected_segments):
    test_file = GeneratedFile()
    for (toc, objects, data) in segments:
        test_file.add_segment(toc, segment_objects_metadata(*objects), data)
    data = test_file.get_bytes_io_file().getvalue()
    tdms_file = TdmsFile.read(BytesIO(data), byte_ranges=True)
    obj = tdms_file[path.split("'")[1]]
    if path.count('/') == 2:
        obj = obj[path.split("'")[3]]

    assert _segment_indexes(tdms_file, obj.byte_ranges()) == expected_segments
    assert_objects_read_from_byte_ranges(data)


def test_byte_ranges_with_one_object_per_segment():
    """ Files where every segment contains the metadata and data of one channel
        only need that channel's segments
    """
    output = BytesIO()
    with TdmsWriter(output) as writer:
        for i in range(30):
            writer.write_segment([ChannelObject('group', 'channel%d' % (i % 3), np.array([i], dtype=np.int32))])
    data = output.getvalue()
    tdms_file = TdmsFile.read(BytesIO(data), byte_ranges=True)
    channel = tdms_file['group']['channel1']

    assert _segment_indexes(tdms_file, channel.byte_ranges()) == list(range(1, 30, 3))
    assert len(channel.byte_ranges()) == 10
    part = TdmsFile.read(BytesIO(extract_segments(BytesIO(data), channel.byte_ranges())))
    np.testing.assert_equal(part['group']['channel1'][:], np.arange(1, 30, 3))
    # The root and group objects are only in the first segment
    assert _segment_indexes(tdms_file, tdms_file.byte_ranges()) == [0]
    assert _segment_indexes(tdms_file, tdms_file['group'].byte_ranges()) == [0]


def test_byte_ranges_of_consecutive_segments_are_merged():
    test_file, _ = scenarios.chunked_segment().values
    data = test_file.get_bytes_io_file().getvalue()
    tdms_file = TdmsFile.read(BytesIO(data), byte_ranges=True)

    assert tdms_file['group']['channel1'].byte_ranges() == [(0, len(data))]


@pytest.mark.parametrize("test_file,expected_data", [
    s for s in scenarios.get_scenarios()
    if s.id not in ('incomplete_last_segment', 'incomplete_last_row_of_interleaved_data')])
def test_byte_ranges_read_from_index_file(test_file, expected_data):
    expected = TdmsFile.read(test_file.get_bytes_io_file(), byte_ranges=True)
    with test_file.get_tempfile_with_index() as tdms_path:
        tdms_file = TdmsFile.read_metadata(tdms_path + '_index', byte_ranges=True)
        assert tdms_file.byte_ranges() == expected.byte_ranges()
        for group in expected.groups():
            assert tdms_file[group.name].byte_ranges() == group.byte_ranges()
            for channel in group.channels():
                assert tdms_file[group.name][channel.name].byte_ranges() == channel.byte_ranges()


def test_byte_ranges_not_recorded():
    tdms_file = TdmsFile.read(scenarios.chunked_segment().values[0].get_bytes_io_file())

    for obj in [tdms_file, tdms_file['group'], tdms_file['group']['channel1']]:
        with pytest.raises(RuntimeError):
            obj.byte_ranges()


def test_merge_byte_ranges():
    assert merge_byte_ranges([(300, 50), (0, 100)], [(100, 100), (0, 100)]) == [(0, 200), (300, 50)]
    assert merge_byte_ranges([(0, 200), (300, 50), (150, 200)]) == [(0, 350)]
    assert merge_byte_ranges() == []


@pytest.mark.parametrize("byte_range,expected", [
    ((0, 5), [[0, 5], [10, 10], [30, 10]]),
    ((5, 5), [[5, 15], [30, 10]]),
    ((12, 3), [[10, 10], [30, 10]]),
    ((15, 20), [[10, 30]]),
    ((0, 50), [[0, 50]]),
    ((45, 5), [[10, 10], [30, 10], [45, 5]]),
])
def test_insert_byte_range(byte_range, expected):
    ranges = [[10, 10], [30, 10]]
    _insert_byte_range(ranges, *byte_range)
    assert ranges == expected


def test_extract_segments_with_wrong_byte_ranges():
    data = scenarios.chunked_segment().values[0].get_bytes_io_file().getvalue()
    byte_ranges = TdmsFile.read(BytesIO(data), byte_ranges=True)['group']['channel1'].byte_ranges()

    with pytest.raises(ValueError):
        extract_segments(BytesIO(data), [(offset + 1, length) for (offset, length) in byte_ranges])
    with pytest.raises(ValueError):
        extract_segments(BytesIO(data[:-1]), byte_ranges)


def test_byte_ranges_stored_in_database():
    test_file, expected_data = scenarios.alternating_data_objects_with_new_obj_list().values
    data = test_file.get_bytes_io_file().getvalue()
    tdms_file = TdmsFile.read_metadata(BytesIO(data), byte_ranges=True)

    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE ranges (path TEXT, offset INTEGER, length INTEGER)")
    for channel in tdms_file['group'].channels():
        db.executemany("INSERT INTO ranges VALUES (?, ?, ?)", [(channel.path, *r) for r in channel.byte_ranges()])

    rows = db.execute("SELECT offset, length FROM ranges WHERE path = ?", ("/'group'/'channel2'", )).fetchall()
    part = TdmsFile.read(BytesIO(extract_segments(BytesIO(data), rows)))
    compare_arrays(part['group']['channel2'][:], expected_data[('group', 'channel2')])


@settings(deadline=None, max_examples=300)
@given(strategies.data())
def test_read_objects_of_random_files_from_byte_ranges(data):
    """ Generate files with random segments that reuse metadata from previous segments
    """
    paths = ["/", "/'group'"] + CHANNELS
    test_file = GeneratedFile()
    object_list = []  # Paths of objects in the current object list, and whether they have data
    num_values = {}  # Number of values in the latest raw data index of each channel
    for i in range(data.draw(strategies.integers(1, 8))):
        kind = data.draw(strategies.sampled_from(["new_list", "incremental"] + (["no_metadata"] if i else [])))
        objects = []
        if kind == "new_list":
            object_list = []
        object_paths = [] if kind == "no_metadata" else data.draw(
            strategies.lists(strategies.sampled_from(paths), unique=True))
        for path in object_paths:
            choices = ["none"] + (["values"] if path in CHANNELS else []) + (["same"] if path in num_values else [])
            index = data.draw(strategies.sampled_from(choices))
            properties = _properties("p", i) if data.draw(strategies.booleans()) else None
            if index == "values":
                num_values[path] = data.draw(strategies.integers(1, 3))
                objects.append(channel_metadata(path, TDS_TYPE_INT32, num_values[path], properties))
            elif index == "same":
                objects.append(channel_metadata_with_repeated_structure(path, properties))
            else:
                objects.append(channel_metadata_with_no_data(path, properties))
            list_paths = [p for (p, _) in object_list]
            if path in list_paths:
                object_list[list_paths.index(path)] = (path, index != "none")
            else:
                object_list.append((path, index != "none"))
        values_per_chunk = [num_values[p] for (p, has_data) in object_list if has_data]
        interleaved = len(set(values_per_chunk)) == 1 and data.draw(strategies.booleans())
        num_chunks = data.draw(strategies.integers(0, 2))
        toc = {"new_list": NEW_LIST, "incremental": INCREMENTAL, "no_metadata": NO_METADATA}[kind]
        test_file.add_segment(
            toc + (("kTocInterleavedData", ) if interleaved else ()),
            "" if kind == "no_metadata" else segment_objects_metadata(*objects),
            _int32_data(*range(num_chunks * sum(values_per_chunk))))
    file_data = test_file.get_bytes_io_file().getvalue()
    try:
        TdmsFile.read(BytesIO(file_data))
    except ValueError:
        assume(False)

    assert_objects_read_from_byte_ranges(file_data)
