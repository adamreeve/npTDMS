"""Range reads must not lose their physical segment position across gaps."""

import struct

import numpy as np
import pytest

from nptdms import TdmsFile
from nptdms.test.util import (
    GeneratedFile, channel_metadata, channel_metadata_with_no_data,
    segment_objects_metadata,
)


@pytest.mark.parametrize("gap_kind", ["absent", "no_data", "zero_values"])
@pytest.mark.parametrize("gap_count", [0, 1, 2])
@pytest.mark.parametrize("initial_gap", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_range_reads_across_channel_gaps(gap_kind, gap_count, initial_gap, interleaved):
    test_file = GeneratedFile()
    expected = []
    target = "/'group'/'target'"
    other = "/'group'/'other'"
    toc = ("kTocMetaData", "kTocRawData", "kTocNewObjList")
    if interleaved:
        toc += ("kTocInterleavedData",)

    def add_gap():
        objects = [channel_metadata(other, 3, 2)]
        if gap_kind == "no_data":
            objects.append(channel_metadata_with_no_data(target))
        elif gap_kind == "zero_values":
            objects.append(channel_metadata(target, 3, 0))
        test_file.add_segment(
            toc, segment_objects_metadata(*objects), struct.pack("<ii", -1, -2).hex())

    if initial_gap:
        add_gap()
    # Multiple chunks in the first and last segments exercise both chunk and
    # segment boundary trimming. The middle segment has a different chunk size.
    for values, chunk_size in [
            ([11, 12, 13, 14], 2), ([21, 22, 23], 3),
            ([31, 32, 33, 34, 35, 36, 37, 38, 39], 3)]:
        if expected:
            for _ in range(gap_count):
                add_gap()
        test_file.add_segment(
            toc, segment_objects_metadata(channel_metadata(target, 3, chunk_size)),
            struct.pack("<" + "i" * len(values), *values).hex())
        expected.extend(values)

    with test_file.get_tempfile() as temp_file:
        with TdmsFile.open(temp_file.file) as tdms_file:
            channel = tdms_file["group"]["target"]
            assert len(channel) == len(expected)
            for offset in range(len(expected) + 2):
                for length in [None, *range(len(expected) + 2)]:
                    stop = None if length is None else offset + length
                    want = np.array(expected[offset:stop], dtype=np.int32)
                    np.testing.assert_array_equal(channel.read_data(offset, length), want)
                    if length is not None:
                        np.testing.assert_array_equal(channel[offset:stop], want)
            np.testing.assert_array_equal(channel[::-1], expected[::-1])
