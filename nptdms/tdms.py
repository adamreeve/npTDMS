"""Python module for reading TDMS files produced by LabView"""

import logging
import struct
import sys
from collections import namedtuple
import numpy as np


log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)
# To adjust the log level for this module from a script, use eg:
# logging.getLogger(tdms.__name__).setLevel(logging.DEBUG)

try:
    long
except NameError:
    # Python 3
    long = int
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
DataType = namedtuple("DataType",
        ('name', 'struct', 'length', 'nptype'))

tdsDataTypes = dict(enumerate((
    DataType('tdsTypeVoid', None, 0, None),
    DataType('tdsTypeI8', 'b', 1, np.int8),
    DataType('tdsTypeI16', 'i', 2, np.int16),
    DataType('tdsTypeI32', 'l', 4, np.int32),
    DataType('tdsTypeI64', 'q', 8, np.int64),
    DataType('tdsTypeU8', 'B', 1, np.uint8),
    DataType('tdsTypeU16', 'I', 2, np.uint16),
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
    0x21: DataType('tdsTypeBoolean', 'b', 8, np.bool8),
    0x44: DataType('tdsTypeTimeStamp', 'Qq', 16, None),
    0xFFFFFFFF: DataType('tdsTypeDAQmxRawData', None, None, None)
})


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
        # Time stamps are stored as number of seconds since the epoch
        # and number of 2^-64 fractions of a second
        s = file.read(data_type.length)
        (s_frac, s) = struct.unpack('%s%s' % (endianness, data_type.struct), s)
        return (s, s_frac * (2 ** -64))
    elif None not in (data_type.struct, data_type.length):
        s = file.read(data_type.length)
        return struct.unpack('%s%s' % (endianness, data_type.struct), s)[0]
    else:
        raise ValueError("Unsupported data type to read, %s." % data_type.name)


class TdmsFile(object):
    """Reads and stores data from a TDMS file.

    :ivar objects: A dictionary of objects in the TDMS file, where the keys are
        the object paths.

    """

    def __init__(self, file):
        """Initialise a new TDMS file object, reading all data.

        :param file: Either the path to the tdms file to read or an already
            opened file.

        """

        self.segments = []
        self.objects = {}

        if hasattr(file, "read"):
            # Is a file
            self._read_segments(file)
        else:
            # Is path to a file
            with open(file, 'rb') as tdms_file:
                self._read_segments(tdms_file)

    def _read_segments(self, tdms_file):
        previous_segment = None
        while True:
            try:
                segment = _TdmsSegment(tdms_file)
            except EOFError:
                # We've finished reading the file
                break
            segment.read_metadata(tdms_file, self.objects,
                    previous_segment)
            segment.read_raw_data(tdms_file, self.objects)

            self.segments.append(segment)
            previous_segment = segment
            if segment.next_segment_pos is None:
                break
            else:
                if(tdms_file.tell() != segment.next_segment_pos):
                    log.warning("Did not read to the end of the "
                            "segment, there may be unread data.")
                    tdms_file.seek(segment.next_segment_pos)

    def _path(self, group=None, channel=None):
        path = '/'
        if group is not None:
            path += "'" + group.replace("'", "''") + "'"
        if channel is not None:
            path += "/'" + channel.replace("'", "''") + "'"
        return path

    def object(self, group, channel=None):
        """Get a TDMS Object from the file

        :param group: Name of the group of the object.
        :keyword channel: The name of the channel, or None to get the
            group object.
        :rtype: :class:`TdmsObject`

        """

        return self.objects[self._path(group, channel)]

    def groups(self):
        """Return the names of groups in the file

        :rtype: List of strings.

        """

        return [path[2:-1]
                for path in self.objects
                if path.count('/') == 1 and path != '/']

    def group_channels(self, group):
        """Returns a list of channel objects for the given group

        :param group: Name of the group to get channels for.
        :rtype: List of :class:`TdmsObject` objects.

        """

        path = self._path(group)
        return [self.objects[p] for p in self.objects if p.startswith(path)]

    def channel_data(self, group, channel):
        """Get the data for a channel

        :param group: The name of the group the channel is in.
        :param channel: The name of the channel to get data for.
        :returns: The channel data.
        :rtype: NumPy array.

        """

        return self.objects[self._path(group, channel)].data


class _TdmsSegment(object):
    def __init__(self, f):
        """Read the lead in section of a segment"""

        self.position = f.tell()
        self.ordered_objects = []

        # First four bytes should be TDSm
        try:
            s = f.read(4).decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Segment does not start with TDSm")
        if s == '':
            raise EOFError
        if s != 'TDSm':
            raise ValueError("Segment does not start with TDSm, "
                    "but with %s" % s)

        log.debug("Reading segment at %d" % self.position)

        # Next four bytes are table of contents mask
        s = f.read(4)
        toc_mask = struct.unpack('<i', s)[0]

        self.toc = {}
        for property in tocProperties.keys():
            self.toc[property] = (toc_mask & tocProperties[property]) != 0
            log.debug("Property %s is %s" % (property, self.toc[property]))

        # Next four bytes are version number
        s = f.read(4)
        self.version = struct.unpack('<i', s)[0]
        if self.version not in (4712, 4713):
            log.warning("Unrecognised version number.")

        # Now 8 bytes each for the offset values
        s = f.read(16)
        (self.next_segment_offset, self.raw_data_offset) = (
                struct.unpack('<QQ', s))

        # Calculate next segment position
        lead_size = 7 * 4
        if self.next_segment_offset == struct.unpack('<Q', b'\xFF' * 8)[0]:
            # This can happen if Labview crashes
            log.warning("Last segment of file has unknown size")
            self.next_segment_pos = None
        else:
            self.next_segment_pos = (self.position +
                    self.next_segment_offset + lead_size)

    def __repr__(self):
        return "<TdmsSegment at position %d>" % self.position

    def read_metadata(self, f, objects, previous_segment=None):
        """Read segment metadata section and update object information"""

        if not self.toc["kTocMetaData"]:
            self.ordered_objects = previous_segment.ordered_objects
            return
        if not self.toc["kTocNewObjList"]:
            self.ordered_objects = previous_segment.ordered_objects

        log.debug("Reading metadata at %d" % f.tell())

        # First four bytes have number of objects in metadata
        s = f.read(4)
        num_objects = struct.unpack("<l", s)[0]

        for obj in range(num_objects):
            # Read the object path
            object_path = read_string(f)

            # If this is a new object, add it to the object dictionary,
            # otherwise reuse the existing object
            if object_path in objects:
                obj = objects[object_path]
            else:
                obj = TdmsObject(object_path)
                objects[object_path] = obj
            obj._read_metadata(f)
            if (self.toc["kTocNewObjList"] or
                    object_path not in [o.path for o in self.ordered_objects]):
                self.ordered_objects.append(obj)

    def read_raw_data(self, f, objects):
        """Read signal data from file"""

        if not self.toc["kTocRawData"]:
            return

        # Work out the number of chunks the data is in, for cases
        # where the meta data doesn't change at all so there is no
        # lead in.
        data_size = sum([
                o.data_size
                for o in self.ordered_objects])
        total_data_size = self.next_segment_offset - self.raw_data_offset
        if total_data_size % data_size != 0:
            raise ValueError("Data size %d is not a multiple of the "
                    "chunk size %d" % (total_data_size, data_size))
        else:
            num_chunks = total_data_size // data_size
        log.debug("Reading %d bytes of data at %d in %d chunks" %
                (total_data_size, f.tell(), num_chunks))

        if self.toc['kTocBigEndian']:
            endianness = '>'
        else:
            endianness = '<'

        for chunk in range(num_chunks):
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
                        object_data[obj.path] = (
                                obj._read_values(f, endianness))

                for obj in self.ordered_objects:
                    if obj.has_data:
                        obj._update_data(object_data[obj.path])

    def _read_interleaved_numpy(self, f, data_objects, endianness):
        """Read interleaved data where all channels have a numpy type"""

        log.debug("Reading interleaved data all at once")
        # Read all data into 1 byte unsigned ints first
        all_channel_bytes = sum((o.data_type.length for o in data_objects))
        number_bytes = all_channel_bytes * data_objects[0].number_values
        combined_data = np.fromfile(f, dtype=np.uint8, count=number_bytes)
        # Reshape, so that one row is all bytes for all objects
        combined_data = combined_data.reshape(-1, all_channel_bytes)
        # Now set arrays for each channel
        data_pos = 0
        for (i, obj) in enumerate(data_objects):
            byte_columns = tuple(
                    range(data_pos, obj.data_type.length + data_pos))
            log.debug("Byte columns for channel %d: %s" % (i, byte_columns))
            # Select columns for this channel, so that number of values will
            # be number of bytes per point * number of data points
            object_data = combined_data[:, byte_columns].ravel()
            # Now set correct data type, so that the array length should
            # be correct
            object_data.dtype = (
                    np.dtype(obj.data_type.nptype).newbyteorder(endianness))
            obj._update_data(object_data)
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
            obj._update_data(object_data[obj.path])


class TdmsObject(object):
    """Represents an object in a TDMS file.

    :ivar path: The TDMS object path.
    :ivar properties: Dictionary of TDMS properties defined for this object,
                      for example the start time and time increment for
                      waveforms.
    :ivar has_data: Boolean, true if there is data associated with the object.
    :ivar data: NumPy array containing data if there is data, otherwise None.

    """

    def __init__(self, path):
        self.path = path
        self.data = None
        self.properties = {}
        self.raw_data_index = 0
        self.data_type = None
        self.dimension = 1
        self.number_values = 0
        self.data_size = 0
        self.has_data = True

    def __repr__(self):
        return "<TdmsObject with path %s>" % self.path

    def _read_metadata(self, f):
        """Read object metadata and update object information"""

        s = f.read(4)
        self.raw_data_index = struct.unpack("<L", s)[0]

        log.debug("Reading metadata for object %s" % self.path)

        # Object has no data in this segment
        if self.raw_data_index == 0xFFFFFFFF:
            self.has_data = False
        # Data has same structure as previously
        elif self.raw_data_index == 0x00000000:
            pass
        else:
            index_length = self.raw_data_index

            # Read the data type
            s = f.read(4)
            self.data_type = tdsDataTypes[struct.unpack("<L", s)[0]]

            # Read data dimension
            s = f.read(4)
            self.dimension = struct.unpack("<L", s)[0]
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
                self.data_size = (self.number_values *
                        self.data_type.length * self.dimension)

            log.debug("Object number of values in segment: %d" %
                    self.number_values)

        # Read data properties
        s = f.read(4)
        num_properties = struct.unpack("<L", s)[0]
        for i in range(num_properties):
            prop_name = read_string(f)

            # Property data type
            s = f.read(4)
            prop_data_type = tdsDataTypes[struct.unpack("<L", s)[0]]
            if prop_data_type.name == 'tdsTypeString':
                value = read_string(f)
            else:
                value = read_type(f, prop_data_type, '<')
            self.properties[prop_name] = value
            log.debug("Property %s: %s" % (prop_name, value))

    def property(self, property_name):
        """Returns the value of a TDMS property

        :param property_name: The name of the property to get.
        :returns: The value of the requested property.
        :raises: KeyError if the property isn't found.

        """

        return self.properties[property_name]

    def time_track(self):
        """Return an array of time for this channel

        This depends on the object having the wf_increment
        and wf_start_offset properties defined.

        :rtype: NumPy array.
        :raises: KeyError if required properties aren't found

        """

        try:
            increment = self.property('wf_increment')
            offset = self.property('wf_start_offset')
        except KeyError:
            raise KeyError("Object does not have time properties available.")

        return np.arange(
                offset, offset + len(self.data) * increment, increment)

    def _new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values

    def _read_value(self, file, endianness):
        """Read a single value from the given file"""

        if self.data_type.nptype is not None:
            dtype = (np.dtype(self.data_type.nptype).
                    newbyteorder(endianness))
            return np.fromfile(file, dtype=dtype, count=1)
        return read_type(file, self.data_type, endianness)

    def _read_values(self, file, endianness):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = (np.dtype(self.data_type.nptype).
                    newbyteorder(endianness))
            return np.fromfile(file, dtype=dtype, count=self.number_values)
        data = self._new_segment_data()
        for i in range(self.number_values):
            data[i] = read_type(file, self.data_type, endianness)
        return data

    def _update_data(self, new_data):
        """Update the object data with a new array of data"""

        log.debug("Adding %d data points to data for %s" %
                (len(new_data), self.path))
        if self.data is None:
            self.data = new_data
        else:
            if self.data_type.nptype is not None:
                self.data = np.append(self.data, new_data)
            else:
                self.data.extend(new_data)
