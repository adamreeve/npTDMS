"""Module for writing TDMS files"""

from collections import OrderedDict
from datetime import datetime
from io import UnsupportedOperation, BytesIO
import os

import numpy as np
from nptdms.common import toc_properties, ObjectPath
from nptdms.types import *
from nptdms import TdmsFile


class TdmsWriter(object):
    """Writes to a TDMS file.

    A TdmsWriter should be used as a context manager, for example::

        with TdmsWriter(path) as tdms_writer:
            tdms_writer.write_segment(segment_data)
    """

    @classmethod
    def defragment(cls, source, destination, version=4712, index_file=False):
        """ Defragemnts an existing TdmsFile by loading and moving each Object to a separate channel
        to stream read one consecutive part of the file for faster access.

        :param source: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        :param destination: Either the path to the tdms file
        :param version: The TDMS format version to write, which must be either 4712 (the default) or 4713.
            It's important that if you are appending segments to an
            existing TDMS file, this matches the existing file version (this can be queried with the
            :py:attr:`~nptdms.TdmsFile.tdms_version` property).
        :param index_file: Depends on the ``destination`` input.
            If ``destination`` is a path ``index_file`` can either be ``True`` or ``False`` to store a ``.tdms_index``
            file at the same folder location or not.
            If ``destination`` is a readable object ``index_file`` can either be a redable object or ``False``
            to store a ``.tdms_index`` file inside of the submitted object or not.
        """
        file = TdmsFile(source)
        with cls(destination, version=version, index_file=index_file) as new_file:
            new_file.write_segment([RootObject(file.properties)])
            for group in file.groups():
                new_file.write_segment([GroupObject(group.name, group.properties)])
                for channel in group.channels():
                    new_file.write_segment([ChannelObject(
                        group.name,
                        channel.name,
                        channel.read_data(scaled=False),
                        channel.properties
                    )])

    def __init__(self, file, mode='w', version=4712, index_file=False):
        """Initialise a new TDMS writer

        :param file: Either the path to the tdms file, an already
            opened file or a bytes stream.
        :param mode: The mode to open the file with, used when ``file`` is a file path.
            This will be passed through to Python's ``open`` function with 'b' appended
            to ensure the file is opened in binary mode.
            For example, use 'w' (the default) to open a new file or 'a' to append to an existing TDMS file.
        :param version: The TDMS format version to write, which must be either 4712 (the default) or 4713.
            It's important that if you are appending segments to an
            existing TDMS file, this matches the existing file version (this can be queried with the
            :py:attr:`~nptdms.TdmsFile.tdms_version` property).
        :param index_file: Whether or not to write a index file besides the data file. Index files
            can be used to accelerate reading speeds for faster channel extraction and data positions inside
            the data files. If ``file```variable is a path ``index_file`` can be ``True`` to store a ``.tdms_index``
            file at the same folder location or ``False`` to only write the data ``.tdms`` file. If ``file`` variable
            is a readable object ``index_file`` can either be a readable object to write into or ``False`` to omit.
        """
        valid_versions = (4712, 4713)
        if version not in valid_versions:
            raise ValueError("version must be one of %s" % ",".join("%d" % v for v in valid_versions))
        self._file = None
        self._index_file = None
        self._file_path = None
        self._index_file_path = None
        self._file_mode = mode
        self._tdms_version = version

        if hasattr(file, "read"):
            # Is a file
            self._file = file
            if hasattr(index_file, "read"):
                self._index_file = index_file
            elif isinstance(index_file, bool) and not index_file:
                pass
            else:
                raise ValueError(
                    f"Invalid type, ``index_file`` can only be ``False`` or a stream to write into, "
                    "but is {type(index_file)}"
                )
        else:
            self._file_path = file
            if isinstance(index_file, bool):
                if index_file:
                    self._index_file_path = file + "_index"
            else:
                raise ValueError(
                    f"Invalid type, ``index_file`` can  only be ``False`` or ``True`` but is {type(index_file)}."
                )

    def open(self):
        if self._file_path is not None:
            self._file = open(self._file_path, self._file_mode + 'b')
            if self._index_file_path is not None:
                self._index_file = open(self._index_file_path, self._file_mode + 'b')

    def close(self):
        if self._file_path is not None:
            self._file.close()

        if self._index_file_path is not None:
            self._index_file.close()

        self._file = None
        self._index_file = None

    def write_segment(self, objects):
        """ Write a segment of data to a TDMS file

        :param objects: A list of TdmsObject instances to write
        """
        segment = TdmsSegment(objects, version=self._tdms_version)
        segment.write(self._file)

        if self._index_file is not None:
            segment = TdmsSegment(objects, is_index_file=True, version=self._tdms_version)
            segment.write(self._index_file)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


class TdmsSegment(object):
    """A segment of data to be written to a file
    """

    def __init__(self, objects, is_index_file=False, version=4712):
        """Initialise a new segment of TDMS data

        :param objects: A list of TdmsObject instances.
        :param is_index_file: Whether a written file is a data file (.tdms) or a index file (.tdms_index).
        :param version: The TDMS format version to write, which must be either 4712 (the default) or 4713.
        """
        paths = set(obj.path for obj in objects)
        if len(paths) != len(objects):
            raise ValueError("Duplicate object paths found")

        self.objects = objects
        self._tdms_version = version
        self.is_index_file = is_index_file

    def write(self, file):
        metadata = self.metadata()
        metadata_size = sum(len(val.bytes) for val in metadata)

        toc = ['kTocMetaData', 'kTocRawData', 'kTocNewObjList']
        leadin = self.leadin(toc, metadata_size)

        file.write(b''.join(val.bytes for val in leadin))
        file.write(b''.join(val.bytes for val in metadata))
        if not self.is_index_file:
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
        if hasattr(obj, 'data'):
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
        leadin.append(Bytes(b'TDSh' if self.is_index_file else b'TDSm'))

        toc_mask = 0
        for toc_flag in toc:
            toc_mask = toc_mask | toc_properties[toc_flag]
        leadin.append(Int32(toc_mask))

        leadin.append(Int32(self._tdms_version))

        next_segment_offset = metadata_size + self._data_size()
        raw_data_offset = metadata_size
        leadin.append(Uint64(next_segment_offset))
        leadin.append(Uint64(raw_data_offset))

        return leadin

    def _data_size(self):
        data_size = 0
        for obj in self.objects:
            if hasattr(obj, 'data'):
                data_size += object_data_size(obj.data_type, obj.data)
        return data_size

    def _write_data(self, file):
        for obj in self.objects:
            if hasattr(obj, 'data'):
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
    """The root TDMS object containing properties for the TDMS file
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
        return str(ObjectPath(self.group))


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
        self.data = _to_np_array(data)
        if self.data.ndim != 1:
            raise ValueError("Channel data must be a 1d array")
        self.properties = properties

    @property
    def has_data(self):
        return True

    @property
    def data_type(self):
        try:
            return numpy_data_types[self.data.dtype]
        except (AttributeError, KeyError):
            try:
                return _to_tdms_value(self.data[0]).__class__
            except IndexError:
                return Void

    @property
    def path(self):
        """The string representation of this channel's path
        """
        return str(ObjectPath(self.group, self.channel))


def read_properties_dict(properties_dict):
    if properties_dict is None:
        return {}

    return OrderedDict(
        (key, _to_tdms_value(val))
        for key, val in properties_dict.items())


def _to_tdms_value(value):
    if isinstance(value, np.number):
        return numpy_data_types[value.dtype](value)
    if isinstance(value, TdmsType):
        return value
    if isinstance(value, bool) or isinstance(value, np.bool_):
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
    if isinstance(value, bytes):
        return String(value)
    raise TypeError("Unsupported property type for %r" % value)


def to_int_property_value(value):
    if value >= 2 ** 63:
        return Uint64(value)
    if value >= 2 ** 31 or value < -2 ** 31:
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
        file.write(array.tobytes())


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


def _to_np_array(data):
    if isinstance(data, np.ndarray):
        return data

    dtype = _infer_dtype(data)
    return np.array(data, dtype=dtype)


def _infer_dtype(data):
    if data and isinstance(data[0], int):
        max_value = max(data)
        min_value = min(data)
        if max_value >= 2**63 and min_value >= 0:
            return np.dtype('uint64')
        elif max_value >= 2**32 or min_value < -1 * 2**31:
            return np.dtype('int64')
        elif max_value >= 2**31 and min_value >= 0:
            return np.dtype('uint32')
        elif max_value >= 2**16 or min_value < -1 * 2**15:
            return np.dtype('int32')
        elif max_value >= 2**15 and min_value >= 0:
            return np.dtype('uint16')
        elif max_value >= 2**8 or min_value < -1 * 2**7:
            return np.dtype('int16')
        elif max_value >= 2**7 and min_value >= 0:
            return np.dtype('uint8')
        else:
            return np.dtype('int8')
    return None
