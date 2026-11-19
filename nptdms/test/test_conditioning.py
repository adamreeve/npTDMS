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


def test_optimize_dtypes():
    noisy = 20.0 + np.random.default_rng(0).normal(0.0, 0.05, 1000)
    # float32 steps are 128 s at this magnitude, and 1010 s at a precision of 1 ms need more than 16 bits
    epoch_seconds = 1.7e9 + np.arange(1000) * 1.0101
    channels = {
        # name: (data, precision, expected raw dtype)
        "flags": (np.array([0.0, 1.0] * 500), None, np.int8),
        "counts": (np.arange(1000.0) - 10.0, None, np.int16),
        "large_counts": (np.arange(1000.0) * 100.0, None, np.float32),
        "single": (noisy.astype(np.float32).astype(np.float64), None, np.float32),
        "noisy": (noisy, None, np.float64),
        "noisy_with_precision": (noisy, 0.01, np.int8),
        # Half of the values are half a step of 2 * precision away from the integers
        "three_decimals": (np.round(1e5 + noisy, 3), 0.001, np.int8),
        "epoch_seconds": (epoch_seconds, 0.001, np.int32),
        "with_nan": (np.array([1.5, np.nan, 2.25]), 0.001, np.float32),
        "ints": (np.arange(10), None, np.int64),
    }
    source = _tdms_file(*(ChannelObject("Group", name, data) for name, (data, _, _) in channels.items()))
    precision = {"/'Group'/'%s'" % name: p for name, (_, p, _) in channels.items()}

    result = _conditioned(source, optimize_dtypes=True, precision=precision)

    for name, (data, precision, raw_dtype) in channels.items():
        channel = result["Group"][name]
        assert channel.raw_data.dtype == raw_dtype, name
        assert channel[:].dtype == (np.float32 if raw_dtype == np.float32 else data.dtype), name
        # Values half a step away can be off by the precision plus float rounding
        atol = precision + 4 * np.spacing(np.nanmax(np.abs(data))) if precision else 0
        np.testing.assert_allclose(channel[:], data, rtol=0, atol=atol, err_msg=name)


@pytest.mark.parametrize("data, precision, warns", [
    (np.array([0.000054, 0.0004, 0.002]), 0.001, True),
    (np.array([0.4, 1.7, 20.3]), 0.5, True),
    (np.array([0.0, 1.0, 2.0]), 0.5, False),
    (np.array([0.000054, 0.0004, 3.2]), 0.001, False),
])
def test_optimize_dtypes_warns_about_coarse_precision(caplog, data, precision, warns):
    _conditioned(_tdms_file(ChannelObject("Group", "small", data)), optimize_dtypes=True, precision=precision)

    assert ("Precision %g is more than 1 %%" % precision in caplog.text) == warns


def test_optimize_dtypes_keeps_scaled_channels():
    scale = {"NI_Scale[0]_Scale_Type": "Linear", "NI_Scale[0]_Linear_Slope": 2.0,
             "NI_Scale[0]_Linear_Y_Intercept": 1.0}
    source = _tdms_file(ChannelObject("Group", "scaled", np.arange(10.0), scale))

    channel = _conditioned(source, optimize_dtypes=True, precision=0.1)["Group"]["scaled"]

    assert channel.raw_data.dtype == np.float64
    np.testing.assert_array_equal(channel[:], np.arange(10.0) * 2.0 + 1.0)


def test_harmonize_time_with_timestamps():
    time = np.datetime64("2026-10-25T12:00:00", "us") + np.arange(100) * np.timedelta64(10, "ms")
    source = _tdms_file(
        ChannelObject("Group", "Time", time),
        ChannelObject("Group", "a", np.arange(100.0)),
        ChannelObject("Group", "b", np.arange(100)))

    group = _conditioned(source, harmonize_time=True)["Group"]

    assert "Time" not in group
    for channel in group.channels():
        assert channel.properties["wf_increment"] == pytest.approx(0.01)
        assert channel.properties["wf_samples"] == 100
        offsets = channel.time_track(absolute_time=True, accuracy="ns") - time.astype("datetime64[ns]")
        np.testing.assert_allclose(offsets.astype(np.int64), 0, atol=1)


def test_harmonize_time_keeps_time_channel_for_other_lengths():
    time = 2.0 + np.arange(10) * 0.5
    source = _tdms_file(
        ChannelObject("Group", "Time", time),
        ChannelObject("Group", "a", np.arange(10.0)),
        ChannelObject("Group", "short", np.arange(5.0)))

    group = _conditioned(source, harmonize_time=True)["Group"]

    np.testing.assert_allclose(group["a"].time_track(), time)
    assert group["a"].properties["wf_start_offset"] == 2.0
    assert "wf_increment" not in group["short"].properties
    np.testing.assert_array_equal(group["Time"][:], time)


def test_harmonize_time_resamples_with_own_time_channels():
    time = np.array([0.0, 0.1, 0.2, 0.35, 0.4, 0.5])
    source = _tdms_file(
        ChannelObject("Group", "p", time * 10.0),
        ChannelObject("Group", "p_Time", time),
        ChannelObject("Group", "s", ["a", "b", "c", "d", "e", "f"]),
        ChannelObject("Group", "s_Time", time),
        ChannelObject("Group", "flag", [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]),
        ChannelObject("Group", "flag_Time", time))

    group = _conditioned(source, harmonize_time=True)["Group"]

    assert sorted(group) == ["flag", "p", "s"]
    assert group["p"].properties["wf_increment"] == pytest.approx(0.1)
    np.testing.assert_allclose(group["p"][:], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert list(group["s"][:]) == ["a", "b", "c", "c", "e", "f"]
    np.testing.assert_array_equal(group["flag"][:], [0.0, 0.0, 1.0, 1.0, 0.0, 1.0])


@pytest.mark.parametrize("options", [
    dict(defragment=False),
])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        condition(DATA_DIR + "/raw_timestamps.tdms", BytesIO(), **options)


def _tdms_file(*channels):
    file = BytesIO()
    with TdmsWriter(file) as writer:
        writer.write_segment(channels)
    file.seek(0)
    return file


def _conditioned(source, **options):
    destination = BytesIO()
    condition(source, destination, **options)
    destination.seek(0)
    return TdmsFile.read(destination)
