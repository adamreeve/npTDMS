"""Python module for writing TDMS files"""

from collections import namedtuple
from io import BytesIO
import logging
import numpy as np
import struct
from nptdms.common import toc_properties, tds_data_types

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


try:
    long
except NameError:
    # Python 3
    long = int


tds_data_type_dict = dict(
    (dt.nptype, dt) for dt in tds_data_types
    if dt.nptype is not None)


class TdmsWriter(object):
    """Writes to a TDMS file.

    A TdmsWriter be used as a context manager, for example::

        with TdmsWriter(path) as tdms_writer:
            tdms_writer.write_segment(segment_data)
    """

    def __init__(self, file):
        """Initialise a new TDMS writer

        :param file: Either the path to the tdms file to open or an already
            opened file.
        """
        self._file = None
        self._file_path = None

        if hasattr(file, "read"):
            # Is a file
            self._file = file
        else:
            self._file_path = file

    def open(self):
        if self._file_path is not None:
            self._file = open(self._file_path, 'wb')

    def close(self):
        if self._file_path is not None:
            self._file.close()
            self._file = None

    def write_segment(self, segments):
        """ Write a segment of data to a TDMS file

        :param segments: A list of ChannelSegment objects to write
        """
        segment = _TdmsSegment(segments)
        segment.write(self._file)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class _TdmsSegment(object):
    """A segment of data to be written to a file
    """

    def __init__(self, channel_segments):
        """Initialise a new segment of TDMS data

        :param channel_segments: A list of ChannelSegment objects.
        """
        self.channel_segments = channel_segments

    def write(self, file):
        metadata = self._metadata()
        data_size = self._data_size()

        toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
        self._write_leadin(file, toc, len(metadata), data_size)

        file.write(metadata)

        self._write_data(file)

    def _metadata(self):
        metadata = []
        metadata.append(struct.pack("<L", len(self.channel_segments)))
        for channel in self.channel_segments:
            metadata.append(string_bytes(
                channel_path(channel.group, channel.channel)))
            channel_index = self._raw_data_index(channel)
            metadata.append(struct.pack('<L', len(channel_index)))
            metadata.append(channel_index)
            #TODO: Write properties
            num_properties = 0
            metadata.append(struct.pack('<L', num_properties))
        return b''.join(metadata)

    def _raw_data_index(self, channel):
        data_type = struct.pack('<l', channel.data_type().enum_value)
        dimension = struct.pack('<L', 1)
        num_values = struct.pack('<Q', len(channel.data))

        return data_type + dimension + num_values

    def _data_size(self):
        data_size = 0
        for channel in self.channel_segments:
            data_size += len(channel.data) * channel.data_type().length
        return data_size

    def _write_leadin(self, file, toc, metadata_size, data_size):
        file.write(b'TDSm')

        toc_mask = long(0)
        for toc_key, toc_val in toc_properties.items():
            if toc_key in toc:
                toc_mask = toc_mask | toc_val
        file.write(struct.pack('<l', toc_mask))

        tdms_version = 4712
        file.write(struct.pack('<l', tdms_version))

        next_segment_offset = metadata_size + data_size
        raw_data_offset = metadata_size

        file.write(struct.pack('<Q', next_segment_offset))
        file.write(struct.pack('<Q', raw_data_offset))

    def _write_data(self, file):
        for channel in self.channel_segments:
            to_file(file, channel.data)


class ChannelSegment(object):
    """A segment of data for a single channel
    """

    def __init__(self, group, channel, data, properties=None):
        """Initialise a new ChannelSegment

        :param group: The name of the group this channel is in.
        :param channel: The name of this channel.
        :param data: 1-D Numpy array of data to be written.
        :param properties: A dictionary mapping property names to
            TdmsValue objects.
        """
        self.group = group
        self.channel = channel
        self.data = data
        self.properties = properties if properties is not None else {}

    def data_type(self):
        return tds_data_type_dict[self.data.dtype.type]


def string_bytes(value):
    """Get bytes representing a string value used in TDMS metadata
    """
    content = value.encode('utf-8')
    length = struct.pack('<L', len(content))
    return length + content


def channel_path(group, channel):
    """Return string representation of object path
    """
    return "/'%s'/'%s'" % (
        group.replace("'", "''"),
        channel.replace("'", "''"))


def to_file(file, array):
    """Wrapper around ndarray.tofile to support BytesIO
    """
    if isinstance(file, BytesIO):
        # tostring actually returns bytes
        file.write(array.tostring())
    else:
        array.tofile(file)
