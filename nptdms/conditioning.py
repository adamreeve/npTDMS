""" Conditioning of TDMS files for a smaller size and faster reading
"""

from nptdms.tdms import TdmsFile
from nptdms.writer import TdmsWriter, RootObject, GroupObject, ChannelObject
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


def condition(
        source, destination, defragment=True, version=None, index_file=False):
    """ Rewrites a TDMS file so it is smaller and faster to read

    :param source: Either the path to the tdms file to read
        as a string or pathlib.Path, or an already opened file.
    :param destination: Either the path to the file to write or an already opened file.
    :param defragment: Write each channel with its properties and all its data to a single segment,
        so a channel can be read from one consecutive part of the file.
        The root object and each group object are written to their own segments.
    :param version: The TDMS format version to write, by default the version of the source file.
    :param index_file: Whether to also write a ``.tdms_index`` file next to ``destination``.
        Only supported when ``destination`` is a path.
    """
    if not defragment:
        raise ValueError("Nothing to do, enable defragment")
    _rewrite(source, destination, version, index_file)


def _rewrite(source, destination, version, index_file):
    with TdmsFile.open(source, raw_timestamps=True) as tdms_file:
        version = tdms_file.tdms_version if version is None else version
        with TdmsWriter(destination, version=version, index_file=index_file) as writer:
            writer.write_segment(RootObject(tdms_file.properties))
            for group in tdms_file.groups():
                writer.write_segment(GroupObject(group.name, group.properties))
                for channel in _conditioned_channels(group):
                    writer.write_segment(channel)


def _conditioned_channels(group):
    """ Yields a ChannelObject with the conditioned data and properties for each channel in a group
    """
    for channel in group.channels():
        data, properties = _channel_data(channel)
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
