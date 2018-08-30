"""Module for writing TDMS files"""

from collections import namedtuple
try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict
from datetime import datetime
from io import UnsupportedOperation
import logging
import numpy as np
from nptdms.common import toc_properties
from nptdms.types import *

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


try:
    long
except NameError:
    # Python 3
    long = int
    unicode = str


class TdmsWriter(object):
    """Writes to a TDMS file.

    A TdmsWriter should be used as a context manager, for example::

        with TdmsWriter(path) as tdms_writer:
            tdms_writer.write_segment(segment_data)
    """

    def __init__(self, file, mode='w'):
        """Initialise a new TDMS writer

        :param file: Either the path to the tdms file to open or an already
            opened file.
        :param mode: Either 'w' to open a new file or 'a' to append to an
            existing TDMS file.
        """
        self._file = None
        self._file_path = None
        self._file_mode = mode

        if hasattr(file, "read"):
            # Is a file
            self._file = file
        else:
            self._file_path = file

    def open(self):
        if self._file_path is not None:
            self._file = open(self._file_path, self._file_mode + 'b')

    def close(self):
        if self._file_path is not None:
            self._file.close()
            self._file = None

    def write_segment(self, objects):
        """ Write a segment of data to a TDMS file

        :param objects: A list of TdmsObject instances to write
        """
        segment = TdmsSegment(objects)
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

        :param objects: A list of TdmsObject instances.
        """
        paths = set(obj.path for obj in objects)
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
            metadata.append(String(obj.path))
            metadata.extend(self.raw_data_index(obj))
            properties = read_properties_dict(obj.properties)
            num_properties = len(properties)
            metadata.append(Uint32(num_properties))
            for prop_name, prop_value in properties.items():
                metadata.append(String(prop_name))
                metadata.append(Int32(prop_value.enum_value))
                metadata.append(prop_value)
        return metadata

    def raw_data_index(self, obj):
        if obj.has_data:
            data_type = Int32(obj.data_type.enum_value)
            dimension = Uint32(1)
            num_values = Uint64(len(obj.data))

            data_index = [Uint32(20), data_type, dimension, num_values]
            # For strings, we also need to write the total data size in bytes
            if obj.data_type == String:
                total_size = object_data_size(obj.data_type, obj.data)
                data_index.append(Uint64(total_size))

            return data_index
        else:
            return [Bytes(b'\xFF\xFF\xFF\xFF')]

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
            if obj.has_data:
                data_size += object_data_size(obj.data_type, obj.data)
        return data_size

    def _write_data(self, file):
        for obj in self.objects:
            if obj.has_data:
                write_data(file, obj)


class TdmsObject(object):
    @property
    def has_data(self):
        return False

    @property
    def data_type(self):
        return None

    @property
    def path(self):
        return None


class RootObject(TdmsObject):
    """The root TDMS object
    """
    def __init__(self, properties=None):
        """Initialise a new GroupObject

        :param properties: A dictionary mapping property names to
            their value.
        """
        self.properties = properties

    @property
    def path(self):
        """The string representation of the root path
        """
        return "/"


class GroupObject(TdmsObject):
    """A TDMS object for a group
    """

    def __init__(self, group, properties=None):
        """Initialise a new GroupObject

        :param group: The name of this group.
        :param properties: A dictionary mapping property names to
            their value.
        """
        self.group = group
        self.properties = properties

    @property
    def path(self):
        """The string representation of this group's path
        """
        return "/'%s'" % self.group.replace("'", "''")


class ChannelObject(TdmsObject):
    """A TDMS object for a channel with data
    """

    def __init__(self, group, channel, data, properties=None):
        """Initialise a new ChannelObject

        :param group: The name of the group this channel is in.
        :param channel: The name of this channel.
        :param data: 1-D Numpy array of data to be written.
        :param properties: A dictionary mapping property names to
            their value.
        """
        self.group = group
        self.channel = channel
        self.data = data
        self.properties = properties

    @property
    def has_data(self):
        return True

    @property
    def data_type(self):
        try:
            return numpy_data_types[self.data.dtype.type]
        except (AttributeError, KeyError):
            try:
                return _to_tdms_value(self.data[0]).__class__
            except IndexError:
                return Void

    @property
    def path(self):
        """The string representation of this channel's path
        """
        return "/'%s'/'%s'" % (
            self.group.replace("'", "''"),
            self.channel.replace("'", "''"))


def read_properties_dict(properties_dict):
    if properties_dict is None:
        return {}

    return OrderedDict(
        (key, _to_tdms_value(val))
        for key, val in properties_dict.items())


def _to_tdms_value(value):
    if isinstance(value, TdmsType):
        return value
    if isinstance(value, bool):
        return Boolean(value)
    if isinstance(value, int):
        return to_int_property_value(value)
    if isinstance(value, float):
        return DoubleFloat(value)
    if isinstance(value, datetime):
        return TimeStamp(value)
    if isinstance(value, np.datetime64):
        return TimeStamp(value)
    if isinstance(value, str):
        return String(value)
    if isinstance(value, unicode):
        return String(value)
    if isinstance(value, bytes):
        return String(value)
    raise TypeError("Unsupported property type for %r" % value)


def to_int_property_value(value):
    if value >= 2 ** 31:
        return Int64(value)
    return Int32(value)


def write_data(file, tdms_object):
    if tdms_object.data_type == TimeStamp:
        # Numpy's datetime format isn't compatible with TDMS,
        # so can't use data.tofile
        write_values(file, tdms_object.data)
    elif tdms_object.data_type == String:
        # Strings are variable size so need to be treated specially
        write_string_values(file, tdms_object.data)
    else:
        try:
            to_file(file, tdms_object.data)
        except (AttributeError):
            # Need to also handle lists of data,
            # to handle timestamp data for example.
            write_values(file, tdms_object.data)


def to_file(file, array):
    """Wrapper around ndarray.tofile to support any file-like object"""

    try:
        array.tofile(file)
    except (TypeError, IOError, UnsupportedOperation):
        # tostring actually returns bytes
        file.write(array.tostring())


def write_values(file, array):
    file.write(b''.join(_to_tdms_value(val).bytes for val in array))


def write_string_values(file, strings):
    try:
        encoded_strings = [s.encode("utf-8") for s in strings]
    except AttributeError:
        # Assume if we can't encode then we already have bytes
        encoded_strings = strings
    offset = 0
    for s in encoded_strings:
        offset += len(s)
        file.write(Uint32(offset).bytes)
    for s in encoded_strings:
        file.write(s)


def object_data_size(data_type, data_values):
    if data_type == String:
        # For string data, the total size is 8 bytes per string for the
        # offsets to the start of each string, plus the length of each string.
        try:
            encoded_strings = [s.encode("utf-8") for s in data_values]
        except AttributeError:
            encoded_strings = data_values
        return sum(4 + len(s) for s in encoded_strings)

    return data_type.size * len(data_values)
