""" Conditioning of TDMS files for a smaller size and faster reading
"""

import os
import shutil
import time
import zipfile
import numpy as np

from nptdms.tdms import TdmsFile
from nptdms.timestamp import TimestampArray
from nptdms.types import Uint32
from nptdms.writer import TdmsWriter, RootObject, GroupObject, ChannelObject, _smallest_int_dtype
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


def condition(
        source, destination, defragment=True, optimize_dtypes=False, precision=None,
        harmonize_time=False, time_channel="Time", time_suffix="_Time",
        archive=False, version=None, index_file=False):
    """ Rewrites a TDMS file so it is smaller and faster to read

    Each step is optional. Optimizing data types or harmonizing time channels always writes the
    defragmented layout.

    :param source: Either the path to the tdms file to read
        as a string or pathlib.Path, or an already opened file.
    :param destination: Either the path to the file to write or an already opened file.
    :param defragment: Write each channel with its properties and all its data to a single segment,
        so a channel can be read from one consecutive part of the file.
        The root object and each group object are written to their own segments.
    :param optimize_dtypes: Store float channels as integers with a linear scale (``NI_Scale`` properties)
        or as single precision floats, where this keeps every value exact or within ``precision``.
        The channels are still read as floats. Integer, boolean, string, timestamp and already scaled channels
        are not changed.
    :param precision: The absolute error allowed when optimizing data types, either a single value
        or a dictionary mapping channel paths to values. Without a precision only exact conversions are made.
        Values smaller than the precision can become 0, so channels with very different magnitudes need
        their own precision. A warning is logged when the precision is more than 1 % of a channel's largest value.
    :param harmonize_time: Convert channels with a time channel into waveform channels with
        ``wf_start_time`` (for timestamps) or ``wf_start_offset`` (for float seconds), ``wf_increment``
        and ``wf_samples`` properties, and drop the time channels that are no longer needed.
        Channels with time values that are not equidistant are resampled with the median time step,
        floats with fractions by linear interpolation and other data, including integer-valued floats
        such as 0/1 flags, with the previous value.
    :param time_channel: Name of a channel that holds the time values for all channels of its group.
    :param time_suffix: Suffix of a channel name that holds the time values for the channel without the suffix,
        for example ``Pressure_Time`` for ``Pressure``. This takes priority over ``time_channel``.
    :param archive: Write ``destination`` as a ZIP archive with a single DEFLATE compressed tdms file,
        which :py:class:`~nptdms.TdmsFile` can read without unpacking it.
    :param version: The TDMS format version to write, by default the version of the source file.
    :param index_file: Whether to also write a ``.tdms_index`` file next to ``destination``.
        Only supported when ``destination`` is a path and ``archive`` is false.
    """
    rewrite = defragment or optimize_dtypes or harmonize_time
    time_names = (time_channel, time_suffix) if harmonize_time else None

    def write(file, index_file=False):
        if rewrite:
            _rewrite(source, file, optimize_dtypes, precision, time_names, version, index_file)
        elif hasattr(source, "read"):
            shutil.copyfileobj(source, file)
        else:
            with open(source, "rb") as source_file:
                shutil.copyfileobj(source_file, file)

    if not archive:
        if not rewrite:
            raise ValueError("Nothing to do, enable defragment, optimize_dtypes, harmonize_time or archive")
        write(destination, index_file)
        return
    if index_file:
        raise ValueError("An archive only holds the tdms file, so index_file must be False")
    member = zipfile.ZipInfo(_member_name(destination), time.localtime()[:6])
    member.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(destination, "w") as archive_file, archive_file.open(member, "w", force_zip64=True) as file:
        write(file)


def _rewrite(source, destination, optimize_dtypes, precision, time_names, version, index_file):
    with TdmsFile.open(source, raw_timestamps=True) as tdms_file:
        version = tdms_file.tdms_version if version is None else version
        with TdmsWriter(destination, version=version, index_file=index_file) as writer:
            writer.write_segment(RootObject(tdms_file.properties))
            for group in tdms_file.groups():
                writer.write_segment(GroupObject(group.name, group.properties))
                for channel in _conditioned_channels(group, optimize_dtypes, precision, time_names):
                    writer.write_segment(channel)


def _conditioned_channels(group, optimize_dtypes, precision, time_names):
    """ Yields a ChannelObject with the conditioned data and properties for each channel in a group
    """
    waveforms, dropped = _plan_waveforms(group, *time_names) if time_names else ({}, set())
    for channel in group.channels():
        if channel.name in dropped:
            continue
        data, properties = _channel_data(channel)
        if channel.name in waveforms:
            time_channel, waveform_properties, resample = waveforms[channel.name]
            properties = {**properties, **waveform_properties}
            if resample:
                data = _resample(data, *_waveform(time_channel)[1])
        if optimize_dtypes and channel._scaling is None and data.dtype.kind == 'f':
            channel_precision = precision.get(channel.path) if isinstance(precision, dict) else precision
            data, scale_properties = _optimized(data, channel_precision, channel.path)
            properties = {**properties, **scale_properties}
        yield ChannelObject(group.name, channel.name, data, properties)


def _channel_data(channel):
    """ The raw data and properties to write for a channel
    """
    data = channel.read_data(scaled=False)
    if not isinstance(data, dict):
        return data, channel.properties
    if len(data) == 1:
        scale_id, data = next(iter(data.items()))
        return data, {**channel.properties, "NI_Scale[%d]_Scale_Type" % scale_id: "AdvancedAPI"}
    # The writer can't write data for several DAQmx scalers
    log.warning("Writing scaled data for channel %s, which has several DAQmx scalers", channel.path)
    return channel.read_data(), {**channel.properties, "NI_Scaling_Status": "scaled"}


def _plan_waveforms(group, time_channel, time_suffix):
    """ Works out which channels of a group become waveforms

    :returns: A dictionary mapping data channel names to their time channel, waveform properties and
        whether they need resampling, and the set of time channel names that are no longer needed.
    """
    time_names = {
        name for name in group
        if name == time_channel or (time_suffix and name.endswith(time_suffix) and name[:-len(time_suffix)] in group)}
    waveforms = {}
    used = set()
    kept = set()
    results = {}
    for channel in group.channels():
        if channel.name in time_names:
            continue
        own_time_channel = channel.name + (time_suffix or "")
        source = own_time_channel if own_time_channel in time_names else time_channel
        if source not in time_names:
            continue
        used.add(source)
        time_channel_object = group[source]
        if source not in results:
            result = _waveform(time_channel_object)
            results[source] = None if result is None else (result[0], result[1] is not None)
        result = results[source]
        if (result is None or len(time_channel_object) != len(channel) or "wf_increment" in channel.properties or
                (result[1] and channel._scaling is not None)):
            log.warning("Channel %s keeps its time channel %s", channel.path, time_channel_object.path)
            kept.add(source)
        else:
            waveforms[channel.name] = (time_channel_object, result[0], result[1])
    return waveforms, used - kept


def _waveform(time_channel):
    """ Waveform properties for the values of a time channel

    :returns: The waveform properties and, if the time values are not equidistant, the time values relative to
        the first one and the times to resample at, or None if the channel can't be used as a time channel.
    """
    if len(time_channel) < 2:
        return None
    time_values = time_channel[:]
    if isinstance(time_values, TimestampArray):
        seconds = ((time_values.seconds - time_values.seconds[0]) +
                   (time_values.second_fractions.astype(np.float64) - float(time_values.second_fractions[0])) *
                   2.0 ** -64)
        properties = {"wf_start_time": time_values[0], "wf_start_offset": 0.0}
    elif time_values.dtype.kind == 'f':
        seconds = time_values - time_values[0]
        properties = {"wf_start_offset": float(time_values[0])}
    else:
        return None
    steps = np.diff(seconds)
    if not (np.isfinite(seconds).all() and (steps > 0).all()):
        return None
    increment = seconds[-1] / (len(seconds) - 1)
    if (np.abs(seconds - np.arange(len(seconds)) * increment) <= 0.01 * increment).all():
        return {**properties, "wf_increment": increment, "wf_samples": len(seconds)}, None
    increment = np.median(steps)
    if steps.max() > 10 * increment:
        log.warning(
            "Time channel %s has gaps of up to %g s, values in gaps are interpolated", time_channel.path, steps.max())
    resampled = np.arange(int(seconds[-1] / increment) + 1) * increment
    return {**properties, "wf_increment": increment, "wf_samples": len(resampled)}, (seconds, resampled)


def _resample(data, seconds, resampled_seconds):
    """ Data at the resampled times, linearly interpolated for floats with fractions
        and the previous value for other data, so integer-valued floats like 0/1 flags keep their values
    """
    if data.dtype.kind == 'f' and not np.array_equal(data, np.round(data)):
        return np.interp(resampled_seconds, seconds, data)
    return data[np.searchsorted(seconds, resampled_seconds, side='right') - 1]


def _optimized(data, precision, path):
    """ Float data with fewer bytes per value, keeping every value exact or within the precision

    :returns: The data to write and the scaling properties to add for it.
    """
    options = [(data, {})]
    with np.errstate(over='ignore'):
        single = data.astype(np.float32)
    if np.allclose(single, data, rtol=0.0, atol=precision or 0.0, equal_nan=True):
        options.append((single, {}))
    options.extend(_scaled_integers(data, precision, path))
    # The first option with the smallest values wins, so ties keep the data as floats
    return min(options, key=lambda option: option[0].dtype.itemsize)


def _scaled_integers(data, precision, path):
    """ Integers with a linear scale that reproduce float data exactly or within the precision

    :returns: A list with a tuple of the integer data and its scaling properties, or an empty list.
    """
    if len(data) == 0 or not np.isfinite(data).all():
        return []
    slopes = [2.0 * precision] if precision else []
    exact = np.array_equal(data, np.round(data))
    if exact:
        slopes.append(1.0)
    if not slopes:
        return []
    slope = max(slopes)
    intercept = np.round((data.min() + data.max()) / 2 / slope) * slope
    raw = np.round((data - intercept) / slope)
    dtype = _smallest_int_dtype(raw.min(), raw.max())
    if dtype is None:
        return []
    raw = raw.astype(dtype)
    largest = np.abs(data).max()
    # Values half a step away are within the precision, apart from float rounding
    atol = precision + 4 * np.spacing(largest) if precision else 0.0
    if not np.allclose(raw * slope + intercept, data, rtol=0.0, atol=atol):
        return []
    if (slope != 1.0 or not exact) and largest < 100 * precision:
        log.warning(
            "Precision %g is more than 1 %% of the largest value %g of channel %s, smaller values can become 0",
            precision, largest, path)
    return [(raw, {
        "NI_Number_Of_Scales": Uint32(1),
        "NI_Scale[0]_Scale_Type": "Linear",
        "NI_Scale[0]_Linear_Slope": slope,
        "NI_Scale[0]_Linear_Y_Intercept": intercept,
        "NI_Scale[0]_Linear_Input_Source": Uint32(0xFFFFFFFF),
    })]


def _member_name(destination):
    """ Name of the tdms file inside an archive, from the archive file name
    """
    name = getattr(destination, "name", destination)
    stem = os.path.splitext(os.path.basename(name))[0] if isinstance(name, (str, os.PathLike)) else "data"
    return stem + ".tdms"
