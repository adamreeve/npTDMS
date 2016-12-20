"""Python module for writing TDMS files"""

from collections import namedtuple
from io import BytesIO
import logging
import numpy as np
from nptdms.common import toc_properties, tds_data_types
from nptdms.value import Int32, Uint32, Uint64, String, Bytes

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
        segment = TdmsSegment(segments)
        segment.write(self._file)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class TdmsSegment(object):
    """A segment of data to be written to a file
    """

    def __init__(self, objects):
        """Initialise a new segment of TDMS data

        :param objects: A list of TdmsObject objects.
        """
        paths = set(obj.path() for obj in objects)
        if len(paths) != len(objects):
            raise ValueError("Duplicate object paths found")

        self.objects = objects

    def write(self, file):
        metadata = self.metadata()
        metadata_size = sum(len(val.bytes) for val in metadata)

        toc = ["kTocMetaData", "kTocRawData", "kTocNewObjList"]
        leadin = self.leadin(toc, metadata_size)

        file.write(b''.join(val.bytes for val in leadin))
        file.write(b''.join(val.bytes for val in metadata))
        self._write_data(file)

    def metadata(self):
        metadata = []
        metadata.append(Uint32(len(self.objects)))
        for obj in self.objects:
            metadata.append(String(obj.path()))
            channel_index = self.raw_data_index(obj)
            metadata.append(Uint32(len(channel_index)))
            metadata.extend(channel_index)
            #TODO: Write properties
            num_properties = 0
            metadata.append(Uint32(num_properties))
        return metadata

    def raw_data_index(self, obj):
        if obj.has_data():
            data_type = Int32(obj.data_type().enum_value)
            dimension = Uint32(1)
            num_values = Uint64(len(obj.data))

            return [data_type, dimension, num_values]
        else:
            return [Int32(0xFFFFFFFF)]

    def leadin(self, toc, metadata_size):
        leadin = []
        leadin.append(Bytes(b'TDSm'))

        toc_mask = long(0)
        for toc_key, toc_val in toc_properties.items():
            if toc_key in toc:
                toc_mask = toc_mask | toc_val
        leadin.append(Int32(toc_mask))

        tdms_version = 4712
        leadin.append(Int32(tdms_version))

        next_segment_offset = metadata_size + self._data_size()
        raw_data_offset = metadata_size
        leadin.append(Uint64(next_segment_offset))
        leadin.append(Uint64(raw_data_offset))

        return leadin

    def _data_size(self):
        data_size = 0
        for obj in self.objects:
            if obj.has_data():
                data_size += len(obj.data) * obj.data_type().length
        return data_size

    def _write_data(self, file):
        for obj in self.objects:
            if obj.has_data():
                to_file(file, obj.data)


class TdmsObject(object):
    def has_data(self):
        return False

    def data_type(self):
        return None

    def path(self):
        return None


class RootObject(TdmsObject):
    def __init__(self, properties=None):
        self.properties = read_properties(properties)

    def path(self):
        return "/"


class GroupObject(TdmsObject):
    def __init__(self, group_name, properties=None):
        self.group = group
        self.properties = read_properties(properties)

    def path(self):
        return "/'%s'" % self.group.replace("'", "''")


class ChannelObject(TdmsObject):
    """A segment of data for a single channel
    """

    def __init__(self, group, channel, data, properties=None):
        """Initialise a new ChannelObject

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

    def has_data(self):
        return True

    def data_type(self):
        return tds_data_type_dict[self.data.dtype.type]

    def path(self):
        """Return string representation of object path
        """
        return "/'%s'/'%s'" % (
            self.group.replace("'", "''"),
            self.channel.replace("'", "''"))


def read_properties(properties_dict):
    return None


def to_file(file, array):
    """Wrapper around ndarray.tofile to support BytesIO
    """
    if isinstance(file, BytesIO):
        # tostring actually returns bytes
        file.write(array.tostring())
    else:
        array.tofile(file)
