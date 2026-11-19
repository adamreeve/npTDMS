""" Tests for conditioning TDMS files """

from io import BytesIO

import numpy as np
import pytest

from nptdms import condition, TdmsFile, TdmsWriter, GroupObject, ChannelObject
from nptdms.test.test_daqmx import daqmx_channel_metadata, daqmx_scaler_metadata, segment_toc
from nptdms.test.test_example_files import DATA_DIR, tdms_files_assert_equal
from nptdms.test.util import GeneratedFile, segment_objects_metadata, root_metadata, group_metadata


def test_defragment_writes_one_segment_per_channel():
    source = BytesIO()
    with TdmsWriter(source, version=4713) as writer:
        for i in range(3):
            writer.write_segment(
                ChannelObject("a", "x", [i, i]), ChannelObject("a", "y", [0.5 * i]), ChannelObject("b", "z", ["s"]))
        writer.write_segment(GroupObject("empty"))
    source.seek(0)

    result = _conditioned(source)

    source.seek(0)
    tdms_files_assert_equal(TdmsFile.read(source), result)
    assert [[o.path for o in segment.ordered_objects] for segment in result._reader._segments] == [
        ["/"], ["/'a'"], ["/'a'/'x'"], ["/'a'/'y'"], ["/'b'"], ["/'b'/'z'"], ["/'empty'"]]
    assert result.tdms_version == 4713


def test_condition_with_index_file(tmp_path):
    path = tmp_path / "conditioned.tdms"
    condition(DATA_DIR + "/raw_timestamps.tdms", path, index_file=True)

    assert (tmp_path / "conditioned.tdms_index").is_file()
    tdms_files_assert_equal(
        TdmsFile.read(DATA_DIR + "/raw_timestamps.tdms", raw_timestamps=True),
        TdmsFile.read(path, raw_timestamps=True))


def test_daqmx_channel_with_several_scalers_is_written_scaled():
    properties = {"NI_Number_Of_Scales": (3, "02 00 00 00")}
    scalers = [daqmx_scaler_metadata(0, 3, 0), daqmx_scaler_metadata(1, 3, 2)]
    metadata = segment_objects_metadata(
        root_metadata(), group_metadata(), daqmx_channel_metadata("Channel1", 4, [4], scalers, properties))
    test_file = GeneratedFile()
    test_file.add_segment(segment_toc(), metadata, "01 00 11 00 02 00 12 00 03 00 13 00 04 00 14 00")

    channel = _conditioned(test_file.get_bytes_io_file())["Group"]["Channel1"]

    np.testing.assert_array_equal(channel[:], [17, 18, 19, 20])
    assert channel.properties["NI_Scaling_Status"] == "scaled"


@pytest.mark.parametrize("options", [
    dict(defragment=False),
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        condition(DATA_DIR + "/raw_timestamps.tdms", BytesIO(), **options)


def _conditioned(source, **options):
    destination = BytesIO()
    condition(source, destination, **options)
    destination.seek(0)
    return TdmsFile.read(destination)
