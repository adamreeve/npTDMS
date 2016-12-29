"""Python module for reading TDMS files produced by LabView"""

import itertools
import logging
import struct
import sys
from collections import namedtuple
try:
    from collections import OrderedDict
except ImportError:
    try:
        # ordereddict available on pypi for Python < 2.7
        from ordereddict import OrderedDict
    except ImportError:
        # Otherwise fall back on normal dict
        OrderedDict = dict
from copy import copy
import numpy as np
from datetime import datetime, timedelta
import tempfile
from io import BytesIO
try:
    import pytz
except ImportError:
    pytz = None

from nptdms.utils import Timer


log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.WARNING)
# To adjust the log level for this module from a script, use eg:
# logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

# Have to get a reference to the builtin property decorator
# so we can use it in TdmsObject, which has a property method.
_property_builtin = property

try:
    long
    zip_longest = itertools.izip_longest
except NameError:
    # Python 3
    long = int
    zip_longest = itertools.zip_longest
tocProperties = {
    'kTocMetaData': (long(1) << 1),
    'kTocRawData': (long(1) << 3),
    'kTocDAQmxRawData': (long(1) << 7),
    'kTocInterleavedData': (long(1) << 5),
    'kTocBigEndian': (long(1) << 6),
    'kTocNewObjList': (long(1) << 2)
}

# Class for describing data types, with data type name,
# identifier used by struct module, the size in bytes to read and the
# numpy data type where applicable/implemented
DataType = namedtuple(
    "DataType", ('name', 'struct', 'length', 'nptype'))

tdsDataTypes = dict(enumerate((
    DataType('tdsTypeVoid', None, 0, None),
    DataType('tdsTypeI8', 'b', 1, np.int8),
    DataType('tdsTypeI16', 'h', 2, np.int16),
    DataType('tdsTypeI32', 'l', 4, np.int32),
    DataType('tdsTypeI64', 'q', 8, np.int64),
    DataType('tdsTypeU8', 'B', 1, np.uint8),
    DataType('tdsTypeU16', 'H', 2, np.uint16),
    DataType('tdsTypeU32', 'L', 4, np.uint32),
    DataType('tdsTypeU64', 'Q', 8, np.uint64),
    DataType('tdsTypeSingleFloat', 'f', 4, np.single),
    DataType('tdsTypeDoubleFloat', 'd', 8, np.double),
    DataType('tdsTypeExtendedFloat', None, None, None),
    DataType('tdsTypeDoubleFloatWithUnit', None, 8, None),
    DataType('tdsTypeExtendedFloatWithUnit', None, None, None)
)))

tdsDataTypes.update({
    0x19: DataType('tdsTypeSingleFloatWithUnit', None, 4, None),
    0x20: DataType('tdsTypeString', None, None, None),
    0x21: DataType('tdsTypeBoolean', 'b', 1, np.bool8),
    0x44: DataType('tdsTypeTimeStamp', 'Qq', 16, None),
    0xFFFFFFFF: DataType('tdsTypeDAQmxRawData', None, 2, np.int16)
})


if pytz:
    # Use UTC time zone if pytz is installed
    timezone = pytz.utc
else:
    timezone = None


def fromfile(file, dtype, count, *args, **kwargs):
    """ Wrapper around np.fromfile to support BytesIO fake files."""

    if isinstance(file, BytesIO):
        return np.fromstring(
            file.read(count * dtype.itemsize),
            dtype=dtype, count=count, *args, **kwargs)
    else:
        return np.fromfile(file, dtype=dtype, count=count, *args, **kwargs)


def read_string(file):
    """Read a string from a tdms file

    For reading strings in the meta data section that start with
    the string length, will not work in the data section"""

    s = file.read(4)
    length = struct.unpack("<L", s)[0]
    return file.read(length).decode('utf-8')


def read_type(file, data_type, endianness):
    """Read a value from the file of the specified data type"""

    if data_type.name == 'tdsTypeTimeStamp':
        # Time stamps are stored as number of seconds since
        # 01/01/1904 00:00:00.00 UTC, ignoring leap seconds,
        # and number of 2^-64 fractions of a second.
        # Note that the TDMS epoch is not the Unix epoch.
        s = file.read(data_type.length)
        (s_frac, s) = struct.unpack('%s%s' % (endianness, data_type.struct), s)
        tdms_start = datetime(1904, 1, 1, 0, 0, 0, tzinfo=timezone)
        ms = float(s_frac) * 5 ** 6 / 2 ** 58
        # Adding timedelta with seconds ignores leap
        # seconds, so this is correct
        return tdms_start + timedelta(seconds=s) + timedelta(microseconds=ms)
    elif None not in (data_type.struct, data_type.length):
        s = file.read(data_type.length)
        return struct.unpack('%s%s' % (endianness, data_type.struct), s)[0]
    else:
        raise ValueError("Unsupported data type to read, %s." % data_type.name)


def read_property(f):
    """ Read a property from a segment's metadata """
    prop_name = read_string(f)

    # Property data type
    s = f.read(4)
    prop_data_type = tdsDataTypes[struct.unpack("<L", s)[0]]
    if prop_data_type.name == 'tdsTypeString':
        value = read_string(f)
    else:
        value = read_type(f, prop_data_type, '<')
    log.debug("Property %s (%s): %s",
              prop_name, prop_data_type.name, value)
    return prop_name, value


# Some simple speed optimisation; discussable if required
_struct_unpack = struct.unpack


def _read_long(a_file):
        t_bytes = a_file.read(4)
        val, = _struct_unpack("<L", t_bytes)
        return val


def _read_long_long(a_file):
        t_bytes = a_file.read(8)
        val, = _struct_unpack("<Q", t_bytes)
        return val


class TdmsFile(object):
    """Reads and stores data from a TDMS file.

    :ivar objects: A dictionary of objects in the TDMS file, where the keys are
        the object paths.

    """

    def __init__(self, file, memmap_dir=None):
        """Initialise a new TDMS file object, reading all data.

        :param file: Either the path to the tdms file to read or an already
            opened file.
        :param memmap_dir: The directory to store memmapped data files in,
            or None to read data into memory. The data files are created
            as temporary files and are deleted when the channel data is no
            longer used. tempfile.gettempdir() can be used to get the default
            temporary file directory.
        """

        self.segments = []
        self.objects = OrderedDict()
        self.memmap_dir = memmap_dir

        if hasattr(file, "read"):
            # Is a file
            self._read_segments(file)
        else:
            # Is path to a file
            with open(file, 'rb') as tdms_file:
                self._read_segments(tdms_file)

    def _read_segments(self, tdms_file):
        with Timer(log, "Read metadata"):
            # Read metadata first to work out how much space we need
            previous_segment = None
            while True:
                try:
                    segment = _TdmsSegment(tdms_file)
                except EOFError:
                    # We've finished reading the file
                    break
                segment.read_metadata(
                    tdms_file, self.objects, previous_segment)

                self.segments.append(segment)
                previous_segment = segment
                if segment.next_segment_pos is None:
                    break
                else:
                    tdms_file.seek(segment.next_segment_pos)

        with Timer(log, "Allocate space"):
            # Allocate space for data
            for object in self.objects.values():
                object._initialise_data(memmap_dir=self.memmap_dir)

        with Timer(log, "Read data"):
            # Now actually read all the data
            for segment in self.segments:
                segment.read_raw_data(tdms_file)

    def _path(self, *args):
        """Convert group and channel to object path"""

        return ('/' + '/'.join(
                ["'" + arg.replace("'", "''") + "'" for arg in args]))

    def object(self, *path):
        """Get a TDMS object from the file

        :param path: The object group and channel. Providing no channel
            returns a group object, and providing no channel or group
            will return the root object.
        :rtype: :class:`TdmsObject`

        For example, to get the root object::

            object()

        To get a group::

            object("group_name")

        To get a channel::

            object("group_name", "channel_name")
        """

        object_path = self._path(*path)
        try:
            return self.objects[object_path]
        except KeyError:
            raise KeyError("Invalid object path: %s" % object_path)

    def groups(self):
        """Return the names of groups in the file

        Note that there is not necessarily a TDMS object associated with
        each group name.

        :rtype: List of strings.

        """

        # Split paths into components and take the first (group) component.
        object_paths = (
            path_components(path)
            for path in self.objects)
        group_names = (path[0] for path in object_paths if len(path) > 0)

        # Use an ordered dict as an ordered set to find unique
        # groups in order.
        groups_set = OrderedDict()
        for group in group_names:
            groups_set[group] = None
        return list(groups_set)

    def group_channels(self, group):
        """Returns a list of channel objects for the given group

        :param group: Name of the group to get channels for.
        :rtype: List of :class:`TdmsObject` objects.

        """

        path = self._path(group)
        return [
            self.objects[p]
            for p in self.objects
            if p.startswith(path + '/')]

    def channel_data(self, group, channel):
        """Get the data for a channel

        :param group: The name of the group the channel is in.
        :param channel: The name of the channel to get data for.
        :returns: The channel data.
        :rtype: NumPy array.

        """

        return self.object(group, channel).data

    def as_dataframe(self, time_index=False, absolute_time=False):
        """
        Converts the TDMS file to a DataFrame

        :param time_index: Whether to include a time index for the dataframe.
        :param absolute_time: If time_index is true, whether the time index
            values are absolute times or relative to the start time.
        :return: The full TDMS file data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        temp = {}
        for key, value in self.objects.items():
            if value.has_data:
                index = value.time_track(absolute_time) if time_index else None
                temp[key] = pd.Series(data=value.data, index=index)
        return pd.DataFrame.from_dict(temp)


class _TdmsSegment(object):

    __slots__ = [
        'position', 'num_chunks', 'ordered_objects', 'toc', 'version',
        'next_segment_offset', 'next_segment_pos',
        'raw_data_offset', 'data_position', 'final_chunk_proportion']

    def __init__(self, f):
        """Read the lead in section of a segment"""

        self.position = f.tell()
        self.num_chunks = 0
        # A list of _TdmsSegmentObject
        self.ordered_objects = []
        self.final_chunk_proportion = 1.0

        # First four bytes should be TDSm
        try:
            s = f.read(4).decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Segment does not start with TDSm")
        if s == '':
            raise EOFError
        if s != 'TDSm':
            raise ValueError(
                "Segment does not start with TDSm, but with %s" % s)

        log.debug("Reading segment at %d", self.position)

        # Next four bytes are table of contents mask
        s = f.read(4)
        toc_mask = struct.unpack('<i', s)[0]

        self.toc = OrderedDict()
        for property in tocProperties.keys():
            self.toc[property] = (toc_mask & tocProperties[property]) != 0
            log.debug("Property %s is %s", property, self.toc[property])

        # Next four bytes are version number
        s = f.read(4)
        self.version = struct.unpack('<i', s)[0]
        if self.version not in (4712, 4713):
            log.warning("Unrecognised version number.")

        # Now 8 bytes each for the offset values
        s = f.read(16)
        (self.next_segment_offset, self.raw_data_offset) = (
            struct.unpack('<QQ', s))

        # Calculate data and next segment position
        lead_size = 7 * 4
        self.data_position = self.position + lead_size + self.raw_data_offset
        if self.next_segment_offset == struct.unpack('<Q', b'\xFF' * 8)[0]:
            # This can happen if Labview crashes
            log.warning(
                "Last segment of file has unknown size, "
                "not attempting to read it")
            self.next_segment_pos = None
            self.next_segment_offset = None
            # Could try to read as much as possible but for now
            # don't attempt to read last segment
            raise EOFError
        else:
            self.next_segment_pos = (
                self.position + self.next_segment_offset + lead_size)

    def __repr__(self):
        return "<TdmsSegment at position %d>" % self.position

    def read_metadata(self, f, objects, previous_segment=None):
        """Read segment metadata section and update object information"""

        if not self.toc["kTocMetaData"]:
            try:
                self.ordered_objects = previous_segment.ordered_objects
            except AttributeError:
                raise ValueError(
                    "kTocMetaData is not set for segment but "
                    "there is no previous segment")
            self.calculate_chunks()
            return
        if not self.toc["kTocNewObjList"]:
            # In this case, there can be a list of new objects that
            # are appended, or previous objects can also be repeated
            # if their properties change
            self.ordered_objects = [
                copy(o) for o in previous_segment.ordered_objects]

        log.debug("Reading metadata at %d", f.tell())

        # First four bytes have number of objects in metadata
        s = f.read(4)
        num_objects = struct.unpack("<l", s)[0]

        for obj in range(num_objects):
            # Read the object path
            object_path = read_string(f)

            # If this is a new segment for an existing object,
            # reuse the existing object, otherwise,
            # create a new object and add it to the object dictionary
            if object_path in objects:
                obj = objects[object_path]
            else:
                obj = TdmsObject(object_path)
                objects[object_path] = obj

            # Add this segment object to the list of segment objects,
            # re-using any properties from previous segments.
            updating_existing = False
            if not self.toc["kTocNewObjList"]:
                # Search for the same object from the previous segment
                # object list.
                obj_index = [
                    i for i, o in enumerate(self.ordered_objects)
                    if o.tdms_object is obj]
                if len(obj_index) > 0:
                    updating_existing = True
                    log.debug("Updating object in segment list")
                    obj_index = obj_index[0]
                    segment_obj = self.ordered_objects[obj_index]
            if not updating_existing:
                if obj._previous_segment_object is not None:
                    log.debug("Copying previous segment object")
                    segment_obj = copy(obj._previous_segment_object)
                else:
                    log.debug("Creating a new segment object")
                    segment_obj = _TdmsSegmentObject(obj)
                self.ordered_objects.append(segment_obj)
            # Read the metadata for this object, updating any
            # data structure information and properties.
            segment_obj._read_metadata(f)
            obj._previous_segment_object = segment_obj

        self.calculate_chunks()

    def calculate_chunks(self):
        """
        Work out the number of chunks the data is in, for cases
        where the meta data doesn't change at all so there is no
        lead in.

        Also increments the number of values for objects in this
        segment, based on the number of chunks.
        """

        if self.toc['kTocDAQmxRawData']:
            # chunks defined differently for DAQmxRawData format
            data_size = self.ordered_objects[0].number_values \
                * self.ordered_objects[0].raw_data_width
        else:
            data_size = sum([
                o.data_size
                for o in self.ordered_objects if o.has_data])

        total_data_size = self.next_segment_offset - self.raw_data_offset
        if data_size < 0 or total_data_size < 0:
            raise ValueError("Negative data size")
        elif data_size == 0:
            # Sometimes kTocRawData is set, but there isn't actually any data
            if total_data_size != data_size:
                raise ValueError(
                    "Zero channel data size but non-zero data "
                    "length based on segment offset.")
            self.num_chunks = 0
            return
        chunk_remainder = total_data_size % data_size
        if chunk_remainder == 0:
            self.num_chunks = int(total_data_size // data_size)

            # Update data count for the overall tdms object
            # using the data count for this segment.
            for obj in self.ordered_objects:
                if obj.has_data:
                    obj.tdms_object.number_values += (
                        obj.number_values * self.num_chunks)

        else:
            log.warning(
                "Data size %d is not a multiple of the "
                "chunk size %d. Will attempt to read last chunk" %
                (total_data_size, data_size))
            self.num_chunks = 1 + int(total_data_size // data_size)

            self.final_chunk_proportion = (
                float(chunk_remainder) / float(data_size))

            for obj in self.ordered_objects:
                if obj.has_data:
                    obj.tdms_object.number_values += (
                        obj.number_values * (self.num_chunks - 1)
                        + int(obj.number_values * self.final_chunk_proportion))

    def read_raw_data(self, f):
        """Read signal data from file"""

        if not self.toc["kTocRawData"]:
            return

        f.seek(self.data_position)

        total_data_size = self.next_segment_offset - self.raw_data_offset
        log.debug(
            "Reading %d bytes of data at %d in %d chunks" %
            (total_data_size, f.tell(), self.num_chunks))

        if self.toc['kTocBigEndian']:
            endianness = '>'
        else:
            endianness = '<'

        for chunk in range(self.num_chunks):
            if self.toc["kTocInterleavedData"]:
                log.debug("Data is interleaved")
                data_objects = [o for o in self.ordered_objects if o.has_data]
                # If all data types have numpy types and all the lengths are
                # the same, then we can read all data at once with numpy,
                # which is much faster
                all_numpy = all(
                    (o.data_type.nptype is not None for o in data_objects))
                same_length = (len(
                    set((o.number_values for o in data_objects))) == 1)
                if (all_numpy and same_length):
                    self._read_interleaved_numpy(f, data_objects, endianness)
                else:
                    self._read_interleaved(f, data_objects, endianness)
            else:
                object_data = {}
                log.debug("Data is contiguous")
                for obj in self.ordered_objects:
                    if obj.has_data:
                        if (chunk == (self.num_chunks - 1) and
                                self.final_chunk_proportion != 1.0):
                            number_values = int(
                                obj.number_values *
                                self.final_chunk_proportion)
                        else:
                            number_values = obj.number_values
                        object_data[obj.path] = (
                            obj._read_values(f, endianness, number_values))

                for obj in self.ordered_objects:
                    if obj.has_data:
                        obj.tdms_object._update_data(object_data[obj.path])

    def _read_interleaved_numpy(self, f, data_objects, endianness):
        """Read interleaved data where all channels have a numpy type"""

        log.debug("Reading interleaved data all at once")
        # Read all data into 1 byte unsigned ints first
        all_channel_bytes = data_objects[0].raw_data_width
        if all_channel_bytes == 0:
            all_channel_bytes = sum((o.data_type.length for o in data_objects))
        log.debug("all_channel_bytes: %d", all_channel_bytes)
        number_bytes = int(all_channel_bytes * data_objects[0].number_values)
        combined_data = fromfile(f, dtype=np.uint8, count=number_bytes)
        # Reshape, so that one row is all bytes for all objects
        combined_data = combined_data.reshape(-1, all_channel_bytes)
        # Now set arrays for each channel
        data_pos = 0
        for (i, obj) in enumerate(data_objects):
            byte_columns = tuple(
                range(data_pos, obj.data_type.length + data_pos))
            log.debug("Byte columns for channel %d: %s", i, byte_columns)
            # Select columns for this channel, so that number of values will
            # be number of bytes per point * number of data points.
            # Then use ravel to flatten the results into a vector.
            object_data = combined_data[:, byte_columns].ravel()
            # Now set correct data type, so that the array length should
            # be correct
            object_data.dtype = (
                np.dtype(obj.data_type.nptype).newbyteorder(endianness))
            obj.tdms_object._update_data(object_data)
            data_pos += obj.data_type.length

    def _read_interleaved(self, f, data_objects, endianness):
        """Read interleaved data that doesn't have a numpy type"""

        log.debug("Reading interleaved data point by point")
        object_data = {}
        points_added = {}
        for obj in data_objects:
            object_data[obj.path] = obj._new_segment_data()
            points_added[obj.path] = 0
        while any([points_added[o.path] < o.number_values
                  for o in data_objects]):
            for obj in data_objects:
                if points_added[obj.path] < obj.number_values:
                    object_data[obj.path][points_added[obj.path]] = (
                        obj._read_value(f, endianness))
                    points_added[obj.path] += 1
        for obj in data_objects:
            obj.tdms_object._update_data(object_data[obj.path])


class TdmsObject(object):
    """Represents an object in a TDMS file.

    :ivar path: The TDMS object path.
    :ivar properties: Dictionary of TDMS properties defined for this object,
                      for example the start time and time increment for
                      waveforms.
    :ivar has_data: Boolean, true if there is data associated with the object.

    """

    def __init__(self, path):
        self.path = path
        self._data = None
        self._data_scaled = None
        self.properties = OrderedDict()
        self.data_type = None
        self.dimension = 1
        self.number_values = 0
        self.has_data = False
        self._previous_segment_object = None

    def __repr__(self):
        return "<TdmsObject with path %s>" % self.path

    def property(self, property_name):
        """Returns the value of a TDMS property

        :param property_name: The name of the property to get.
        :returns: The value of the requested property.
        :raises: KeyError if the property isn't found.

        """

        try:
            return self.properties[property_name]
        except KeyError:
            raise KeyError(
                "Object does not have property '%s'" % property_name)

    @_property_builtin
    def group(self):
        """ Returns the name of the group for this object,
            or None if it is the root object.
        """
        path = path_components(self.path)
        if len(path) > 0:
            return path[0]
        return None

    @_property_builtin
    def channel(self):
        """ Returns the name of the channel for this object,
            or None if it is a group or the root object.
        """
        path = path_components(self.path)
        if len(path) > 1:
            return path[1]
        return None

    def time_track(self, absolute_time=False, accuracy='ns'):
        """Return an array of time or the independent variable for this channel

        This depends on the object having the wf_increment
        and wf_start_offset properties defined.
        Note that wf_start_offset is usually zero for time-series data.
        If you have time-series data channels with different start times,
        you should use the absolute time or calculate the time offsets using
        the wf_start_time property.

        For larger timespans, the accuracy setting should be set lower.
        The default setting is 'ns', which has a timespan of
        [1678 AD, 2262 AD]. For the exact ranges, refer to
            http://docs.scipy.org/doc/numpy/reference/arrays.datetime.html
        section "Datetime Units".

        :param absolute_time: Whether the returned time values are absolute
            times rather than relative to the start time.
        :param accuracy: The accuracy of the returned datetime64 array.
        :rtype: NumPy array.
        :raises: KeyError if required properties aren't found

        """

        try:
            increment = self.property('wf_increment')
            offset = self.property('wf_start_offset')
        except KeyError:
            raise KeyError("Object does not have time properties available.")

        periods = len(self._data)

        relative_time = np.linspace(
            offset,
            offset + (periods - 1) * increment,
            periods)

        if not absolute_time:
            return relative_time

        try:
            start_time = self.property('wf_start_time')
        except KeyError:
            raise KeyError(
                "Object does not have start time property available.")

        try:
            unit_correction = {
                's': 1e0,
                'ms': 1e3,
                'us': 1e6,
                'ns': 1e9,
            }[accuracy]
        except KeyError:
            raise KeyError("Invalid accuracy: {0}".format(accuracy))

        # Because numpy only knows ints as its date datatype,
        # convert to accuracy.
        time_type = "timedelta64[{0}]".format(accuracy)
        return (np.datetime64(start_time)
                + (relative_time * unit_correction).astype(time_type))

    def _initialise_data(self, memmap_dir=None):
        """Initialise data array to zeros"""

        if self.number_values == 0:
            pass
        elif self.data_type.nptype is None:
            self._data = []
        else:
            if memmap_dir:
                memmap_file = tempfile.NamedTemporaryFile(
                    mode='w+b', prefix="nptdms_", dir=memmap_dir)
                self._data = np.memmap(
                    memmap_file.file,
                    mode='w+',
                    shape=(self.number_values,),
                    dtype=self.data_type.nptype)
            else:
                self._data = np.zeros(
                    self.number_values, dtype=self.data_type.nptype)
            self._data_insert_position = 0
        if self._data is not None:
            log.debug("Allocated %d sample slots for %s", len(self._data),
                      self.path)
        else:
            log.debug("Allocated no space for %s", self.path)

    def _update_data(self, new_data):
        """Update the object data with a new array of data"""

        log.debug("Adding %d data points to data for %s" %
                  (len(new_data), self.path))
        if self._data is None:
            self._data = new_data
        else:
            if self.data_type.nptype is not None:
                data_pos = (
                    self._data_insert_position,
                    self._data_insert_position + len(new_data))
                self._data_insert_position += len(new_data)
                self._data[data_pos[0]:data_pos[1]] = new_data
            else:
                self._data.extend(new_data)

    def as_dataframe(self, absolute_time=False):
        """
        Converts the TDMS object to a DataFrame

        :param absolute_time: Whether times should be absolute rather than
            relative to the start time.
        :return: The TDMS object data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        # When absolute_time is True,
        # use the wf_start_time as offset for the time_track()
        time = self.time_track(absolute_time)

        return pd.DataFrame(self._data, index=time, columns=[self.path])

    @_property_builtin
    def data(self):
        """
        NumPy array containing data if there is data for this object,
        otherwise None.
        """
        if self._data is None:
            # self._data is None if data segment is empty
            return np.empty((0, 1))
        if self._data_scaled is None:
            scale_type = self.properties.get('NI_Scale[1]_Scale_Type', None)
            if scale_type == 'Polynomial':
                coeff_names = ['NI_Scale[1]_Polynomial_Coefficients[%d]' % i
                               for i in range(4)]
                scaled_data = np.zeros_like(self._data, dtype=np.float)
                for i, scale_factor in enumerate([self.properties[s]
                                                  for s in coeff_names]):
                    scaled_data += scale_factor * self._data**i
                self._data_scaled = scaled_data
            elif scale_type == 'Linear':
                slope = self.properties["NI_Scale[1]_Linear_Slope"]
                intercept = self.properties["NI_Scale[1]_Linear_Y_Intercept"]
                self._data_scaled = self._data * slope + intercept
            else:
                self._data_scaled = self._data

        return self._data_scaled

    @_property_builtin
    def raw_data(self):
        """
        For objects that contain DAQmx raw data, this is the raw, unscaled data
        array. For other objects this is the same as the data property.
        """
        if self._data is None:
            # self._data is None if data segment is empty
            return np.empty((0, 1))
        return self._data


class _TdmsmxDAQMetadata(object):
    __slots__ = [
        'chunk_size',
        'data_type',
        'dimension',
        'raw_data_widths',
        'scale_id',
        'scaler_data_type',
        'scaler_data_type_code',
        'scaler_raw_buffer_index',
        'scaler_raw_buffer_index',
        'scaler_raw_byte_offset',
        'scaler_sample_format_bitmap',
        'scaler_vector_length',
        ]

    def info(self):
        l = []
        for name in self.__slots__:
            l.append("%s: %s" % (name, getattr(self, name)))

        tmp = ",".join(l)
        fmt = "%s: ('%s')"
        txt = fmt % (self.__class__.__name__, tmp)
        return txt

    def _read_metadata(self, f):
        """
        Read the metadata for a DAQmx raw segment.  This is the raw
        DAQmx-specific portion of the raw data index.
        """
        self.data_type = tdsDataTypes[0xFFFFFFFF]
        self.dimension = _read_long(f)
        # In TDMS format version 2.0, 1 is the only valid value for dimension
        if self.dimension != 1:
            log.warning("Data dimension is not 1")
        self.chunk_size = _read_long_long(f)
        # size of vector of format changing scalers
        self.scaler_vector_length = _read_long(f)
        # Size of the vector
        log.debug("mxDAQ format scaler vector size '%d'" %
                  (self.scaler_vector_length,))
        if self.scaler_vector_length > 1:
            log.error("mxDAQ multiple format changing scalers not implemented")

        for idx in range(self.scaler_vector_length):
            # WARNING: This code overwrites previous values with new
            # values.  At this time NI provides no documentation on
            # how to use these scalers and sample TDMS files do not
            # include more than one of these scalers.
            self.scaler_data_type_code = _read_long(f)
            self.scaler_data_type = tdsDataTypes[self.scaler_data_type_code]

            # more info for format changing scaler
            self.scaler_raw_buffer_index = _read_long(f)
            self.scaler_raw_byte_offset = _read_long(f)
            self.scaler_sample_format_bitmap = _read_long(f)
            self.scale_id = _read_long(f)

        raw_data_widths_length = _read_long(f)
        self.raw_data_widths = np.zeros(raw_data_widths_length, dtype=np.int32)
        for cnt in range(raw_data_widths_length):
            self.raw_data_widths[cnt] = _read_long(f)


class _TdmsSegmentObject(object):
    """
    Describes an object in an individual TDMS file segment
    """

    __slots__ = [
        'tdms_object', 'number_values', 'data_size',
        'has_data', 'data_type', 'dimension',
        'raw_data_width']

    def __init__(self, tdms_object):
        self.tdms_object = tdms_object

        self.number_values = 0
        self.data_size = 0
        self.has_data = True
        self.data_type = None
        self.dimension = 1
        self.raw_data_width = 0

    def _read_metadata_mx(self, f):

        # Read the data type
        s = f.read(4)

        data_type_val, = struct.unpack("<L", s)
        try:
            self.data_type = tdsDataTypes[data_type_val]
        except KeyError:
            raise KeyError("Unrecognised data type")

        if self.tdms_object.data_type is not None \
           and self.data_type != self.tdms_object.data_type:

            raise ValueError(
                "Segment object doesn't have the same data "
                "type as previous segments.")
        else:
            self.tdms_object.data_type = self.data_type

        log.debug("mxDAQ Object data type: %s",
                  self.tdms_object.data_type.name)

        info = _TdmsmxDAQMetadata()
        info._read_metadata(f)

        log.debug("mxDAQ '%s' '%s'", info, info.info())
        return info

    def _read_metadata(self, f):
        """Read object metadata and update object information"""

        s = f.read(4)
        raw_data_index = struct.unpack("<L", s)[0]

        log.debug("Reading metadata for object %s", self.tdms_object.path)

        # Object has no data in this segment
        if raw_data_index == 0xFFFFFFFF:
            log.debug("Object has no data in this segment")
            self.has_data = False
            # Leave number_values and data_size as set previously,
            # as these may be re-used by later segments.
        # Data has same structure as previously
        elif raw_data_index == 0x00000000:
            log.debug(
                "Object has same data structure as in the previous segment")
            self.has_data = True
        elif raw_data_index == 0x00001269 or raw_data_index == 0x00001369:
            # This is a DAQmx raw data segment.
            #    0x00001269 for segment containing Format Changing scaler.
            #    0x00001369 for segment containing Digital Line scaler.
            if raw_data_index == 0x00001369:
                # special scaling for DAQ's digital input lines?
                log.warning("DAQmx with Digital Line scaler has not tested")

            # DAQmx raw data format metadata has its own class
            self.has_data = True
            self.tdms_object.has_data = True

            info = self._read_metadata_mx(f)
            self.dimension = info.dimension
            self.data_type = info.data_type
            # DAQmx format has special chunking
            self.data_size = info.chunk_size
            self.number_values = info.chunk_size/info.data_type.length
            # segment reading code relies on a single consistent raw
            # data width so assert that there is only one.
            assert(len(info.raw_data_widths) == 1)
            self.raw_data_width = info.raw_data_widths[0]
            # fall through and read properties
        else:
            # Assume metadata format is legacy TDMS format.
            # raw_data_index gives the length of the index information.
            self.has_data = True
            self.tdms_object.has_data = True

            # Read the data type
            s = f.read(4)
            try:
                self.data_type = tdsDataTypes[struct.unpack("<L", s)[0]]
            except KeyError:
                raise KeyError("Unrecognised data type")
            if (self.tdms_object.data_type is not None and
                    self.data_type != self.tdms_object.data_type):
                raise ValueError(
                    "Segment object doesn't have the same data "
                    "type as previous segments.")
            else:
                self.tdms_object.data_type = self.data_type
            log.debug("Object data type: %s", self.tdms_object.data_type.name)

            if (self.tdms_object.data_type.length is None and
                    self.tdms_object.data_type.name != 'tdsTypeString'):
                raise ValueError(
                    "Unsupported data type: %s" %
                    self.tdms_object.data_type.name)

            # Read data dimension
            s = f.read(4)
            self.dimension = struct.unpack("<L", s)[0]
            # In TDMS version 2.0, 1 is the only valid value for dimension
            if self.dimension != 1:
                log.warning("Data dimension is not 1")

            # Read number of values
            s = f.read(8)
            self.number_values = struct.unpack("<Q", s)[0]

            # Variable length data types have total size
            if self.data_type.name in ('tdsTypeString', ):
                s = f.read(8)
                self.data_size = struct.unpack("<Q", s)[0]
            else:
                self.data_size = (
                    self.number_values *
                    self.data_type.length * self.dimension)

            log.debug(
                "Object number of values in segment: %d", self.number_values)

        # Read data properties
        s = f.read(4)
        num_properties = struct.unpack("<L", s)[0]
        log.debug("Reading %d properties", num_properties)
        for i in range(num_properties):
            prop_name, value = read_property(f)
            self.tdms_object.properties[prop_name] = value

    @property
    def path(self):
        return self.tdms_object.path

    def _read_value(self, file, endianness):
        """Read a single value from the given file"""

        if self.data_type.nptype is not None:
            dtype = (np.dtype(self.data_type.nptype).newbyteorder(endianness))
            return fromfile(file, dtype=dtype, count=1)
        return read_type(file, self.data_type, endianness)

    def _read_values(self, file, endianness, number_values):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = (np.dtype(self.data_type.nptype).newbyteorder(endianness))
            return fromfile(file, dtype=dtype, count=number_values)
        elif self.data_type.name == "tdsTypeString":
            return read_string_data(file, number_values)
        data = self._new_segment_data()
        for i in range(number_values):
            data[i] = read_type(file, self.data_type, endianness)
        return data

    def _new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values


def read_string_data(file, number_values):
    """ Read string raw data

        This is stored as an array of offsets
        followed by the contiguous string data.
    """
    offsets = [0]
    for i in range(number_values):
        s = file.read(4)
        offset = struct.unpack("<L", s)[0]
        offsets.append(offset)
    strings = []
    for i in range(number_values):
        s = file.read(offsets[i + 1] - offsets[i])
        strings.append(s.decode('utf-8'))
    return strings


def path_components(path):
    """Convert a path into group and channel name components"""

    def yield_components(path):
        # Iterate over each character and the next character
        chars = zip_longest(path, path[1:])
        try:
            # Iterate over components
            while True:
                c, n = next(chars)
                if c != '/':
                    raise ValueError("Invalid path, expected \"/\"")
                elif (n is not None and n != "'"):
                    raise ValueError("Invalid path, expected \"'\"")
                else:
                    # Consume "'" or raise StopIteration if at the end
                    next(chars)
                component = []
                # Iterate over characters in component name
                while True:
                    c, n = next(chars)
                    if c == "'" and n == "'":
                        component += "'"
                        # Consume second "'"
                        next(chars)
                    elif c == "'":
                        yield "".join(component)
                        break
                    else:
                        component += c
        except StopIteration:
            return

    return list(yield_components(path))
