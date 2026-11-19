""" Conditioning of TDMS files for a smaller size and faster reading
"""

import numpy as np

from nptdms.tdms import TdmsFile
from nptdms.types import Uint32
from nptdms.writer import TdmsWriter, RootObject, GroupObject, ChannelObject, _smallest_int_dtype
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


def condition(
        source, destination, defragment=True, optimize_dtypes=False, precision=None,
        version=None, index_file=False):
    """ Rewrites a TDMS file so it is smaller and faster to read

    Each step is optional. Optimizing data types always writes the defragmented layout.

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
    :param version: The TDMS format version to write, by default the version of the source file.
    :param index_file: Whether to also write a ``.tdms_index`` file next to ``destination``.
        Only supported when ``destination`` is a path.
    """
    if not (defragment or optimize_dtypes):
        raise ValueError("Nothing to do, enable defragment or optimize_dtypes")
    _rewrite(source, destination, optimize_dtypes, precision, version, index_file)


def _rewrite(source, destination, optimize_dtypes, precision, version, index_file):
    with TdmsFile.open(source, raw_timestamps=True) as tdms_file:
        version = tdms_file.tdms_version if version is None else version
        with TdmsWriter(destination, version=version, index_file=index_file) as writer:
            writer.write_segment(RootObject(tdms_file.properties))
            for group in tdms_file.groups():
                writer.write_segment(GroupObject(group.name, group.properties))
                for channel in _conditioned_channels(group, optimize_dtypes, precision):
                    writer.write_segment(channel)


def _conditioned_channels(group, optimize_dtypes, precision):
    """ Yields a ChannelObject with the conditioned data and properties for each channel in a group
    """
    for channel in group.channels():
        data, properties = _channel_data(channel)
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
