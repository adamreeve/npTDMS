from copy import copy
from io import UnsupportedOperation
import tempfile
import os
import numpy as np

from nptdms import scaling
from nptdms import types
from nptdms.common import toc_properties, path_components
from nptdms.utils import OrderedDict
from nptdms.log import log_manager
from nptdms.daqmx import DaqMxMetadata


# Have to get a reference to the builtin property decorator
# so we can use it in TdmsObject, which has a property method.
_property_builtin = property


log = log_manager.get_logger(__name__)


class TdmsSegment(object):

    __slots__ = [
        'position', 'num_chunks', 'ordered_objects', 'toc', 'version',
        'next_segment_offset', 'next_segment_pos', 'tdms_file',
        'raw_data_offset', 'data_position', 'final_chunk_proportion',
        'endianness']

    def __init__(self, f, tdms_file):
        """Read the lead in section of a segment"""

        self.tdms_file = tdms_file
        self.position = f.tell()
        self.num_chunks = 0
        self.endianness = "<"
        # A list of TdmsSegmentObject
        self.ordered_objects = []
        self.final_chunk_proportion = 1.0

        # First four bytes should be TDSm
        try:
            tag = f.read(4).decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError("Segment does not start with TDSm")
        if tag == '':
            raise EOFError
        if tag != 'TDSm':
            raise ValueError(
                "Segment does not start with TDSm, but with %s" % tag)

        log.debug("Reading segment at %d", self.position)

        # Next four bytes are table of contents mask
        toc_mask = types.Int32.read(f)

        self.toc = OrderedDict()
        for prop_name, prop_mask in toc_properties.items():
            prop_is_set = (toc_mask & prop_mask) != 0
            self.toc[prop_name] = prop_is_set
            log.debug("Property %s is %s", prop_name, prop_is_set)

        if self.toc['kTocBigEndian']:
            self.endianness = '>'

        # Next four bytes are version number
        self.version = types.Int32.read(f, self.endianness)
        if self.version not in (4712, 4713):
            log.warning("Unrecognised version number.")

        # Now 8 bytes each for the offset values
        self.next_segment_offset = types.Uint64.read(f, self.endianness)
        self.raw_data_offset = types.Uint64.read(f, self.endianness)

        # Calculate data and next segment position
        lead_size = 7 * 4
        self.data_position = self.position + lead_size + self.raw_data_offset
        if self.next_segment_offset == 0xFFFFFFFFFFFFFFFF:
            # Segment size is unknown. This can happen if Labview crashes.
            # Try to read until the end of the file.
            log.warning(
                "Last segment of file has unknown size, "
                "will attempt to read to the end of the file")
            current_pos = f.tell()
            f.seek(0, os.SEEK_END)
            end_pos = f.tell()
            f.seek(current_pos, os.SEEK_SET)

            self.next_segment_pos = end_pos
            self.next_segment_offset = end_pos - self.position - lead_size
        else:
            log.debug("Next segment offset = %d, raw data offset = %d",
                      self.next_segment_offset, self.raw_data_offset)
            log.debug("Data size = %d b",
                      self.next_segment_offset - self.raw_data_offset)
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
        num_objects = types.Int32.read(f, self.endianness)

        for obj in range(num_objects):
            # Read the object path
            object_path = types.String.read(f, self.endianness)

            # If this is a new segment for an existing object,
            # reuse the existing object, otherwise,
            # create a new object and add it to the object dictionary
            if object_path in objects:
                obj = objects[object_path]
            else:
                obj = TdmsObject(object_path, self.tdms_file)
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
                    log.debug("Copying previous segment object for %s",
                              object_path)
                    segment_obj = copy(obj._previous_segment_object)
                else:
                    log.debug("Creating a new segment object for %s",
                              object_path)
                    segment_obj = TdmsSegmentObject(obj, self.endianness)
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
            try:
                data_size = next(
                    o.number_values * o.total_raw_data_width
                    for o in self.ordered_objects
                    if o.has_data and
                    o.number_values * o.total_raw_data_width > 0)
            except StopIteration:
                data_size = 0
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
                    "Zero channel data size but data length based on "
                    "segment offset is %d." % total_data_size)
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
                "chunk size %d. Will attempt to read last chunk",
                total_data_size, data_size)
            self.num_chunks = 1 + int(total_data_size // data_size)

            self.final_chunk_proportion = (
                float(chunk_remainder) / float(data_size))

            for obj in self.ordered_objects:
                if obj.has_data:
                    obj.tdms_object.number_values += (
                        obj.number_values * (self.num_chunks - 1) + int(
                            obj.number_values * self.final_chunk_proportion))

    def read_raw_data(self, f):
        """Read signal data from file"""

        if not self.toc["kTocRawData"]:
            return

        f.seek(self.data_position)

        total_data_size = self.next_segment_offset - self.raw_data_offset
        log.debug(
            "Reading %d bytes of data at %d in %d chunks",
            total_data_size, f.tell(), self.num_chunks)

        for chunk in range(self.num_chunks):
            if self.toc['kTocDAQmxRawData']:
                data_objects = [o for o in self.ordered_objects if o.has_data]
                self._read_interleaved_daqmx(f, data_objects)
            elif self.toc["kTocInterleavedData"]:
                log.debug("Data is interleaved")
                data_objects = [o for o in self.ordered_objects if o.has_data]
                # If all data types have numpy types and all the lengths are
                # the same, then we can read all data at once with numpy,
                # which is much faster
                all_numpy = all(
                    o.data_type.nptype is not None for o in data_objects)
                same_length = (len(
                    set((o.number_values for o in data_objects))) == 1)
                if (all_numpy and same_length):
                    self._read_interleaved_numpy(f, data_objects)
                else:
                    self._read_interleaved(f, data_objects)
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
                            obj._read_values(f, number_values))

                for obj in self.ordered_objects:
                    if obj.has_data:
                        obj.tdms_object._update_data(object_data[obj.path])

    def _read_interleaved_daqmx(self, f, data_objects):
        """Read data from DAQmx data segment"""

        log.debug("Reading DAQmx data segment")

        # If we have DAQmx data, we expect all objects to have matching
        # raw data widths, and this gives the number of bytes to read:
        all_daqmx = all(
            o.data_type == types.DaqMxRawData for o in data_objects)
        if not all_daqmx:
            raise Exception("Cannot read a mix of DAQmx and "
                            "non-DAQmx interleaved data")
        all_channel_bytes = data_objects[0].total_raw_data_width

        log.debug("all_channel_bytes: %d", all_channel_bytes)
        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            f, all_channel_bytes, data_objects[0].number_values)

        # Now set arrays for each scaler of each channel
        for (i, obj) in enumerate(data_objects):
            for scaler in obj.daqmx_metadata.scalers:
                offset = scaler.raw_byte_offset
                scaler_size = scaler.data_type.size
                byte_columns = tuple(
                    range(offset, offset + scaler_size))
                log.debug("Byte columns for channel %d scaler %d: %s",
                          i, scaler.scale_id, byte_columns)
                # Select columns for this scaler, so that number of values will
                # be number of bytes per point * number of data points.
                # Then use ravel to flatten the results into a vector.
                object_data = combined_data[:, byte_columns].ravel()
                # Now set correct data type, so that the array length should
                # be correct
                object_data.dtype = (
                    scaler.data_type.nptype.newbyteorder(self.endianness))
                obj.tdms_object._update_data_for_scaler(
                    scaler.scale_id, object_data)

    def _read_interleaved_numpy(self, f, data_objects):
        """Read interleaved data where all channels have a numpy type"""

        log.debug("Reading interleaved data all at once")

        # For non-DAQmx, simply use the data type sizes
        all_channel_bytes = sum(o.data_type.size for o in data_objects)
        log.debug("all_channel_bytes: %d", all_channel_bytes)

        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            f, all_channel_bytes, data_objects[0].number_values)

        # Now set arrays for each channel
        data_pos = 0
        for (i, obj) in enumerate(data_objects):
            byte_columns = tuple(
                range(data_pos, obj.data_type.size + data_pos))
            log.debug("Byte columns for channel %d: %s", i, byte_columns)
            # Select columns for this channel, so that number of values will
            # be number of bytes per point * number of data points.
            # Then use ravel to flatten the results into a vector.
            object_data = combined_data[:, byte_columns].ravel()
            # Now set correct data type, so that the array length should
            # be correct
            object_data.dtype = (
                obj.data_type.nptype.newbyteorder(self.endianness))
            obj.tdms_object._update_data(object_data)
            data_pos += obj.data_type.size

    def _read_interleaved(self, f, data_objects):
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
                        obj._read_value(f))
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

    def __init__(self, path, tdms_file=None):
        self.path = path
        self.tdms_file = tdms_file
        self._data = None
        self._scaler_data = {}
        self._data_scaled = None
        self.properties = OrderedDict()
        self.data_type = None
        self.dimension = 1
        self.number_values = 0
        self.has_data = False
        self._previous_segment_object = None
        self._data_insert_position = 0
        self._scaler_insert_positions = {}

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
            times rather than relative to the start time. If true, the
            wf_start_time property must be set.
        :param accuracy: The accuracy of the returned datetime64 array.
        :rtype: NumPy array.
        :raises: KeyError if required properties aren't found

        """

        try:
            increment = self.property('wf_increment')
            offset = self.property('wf_start_offset')
        except KeyError:
            raise KeyError("Object does not have time properties available.")

        relative_time = np.linspace(
            offset,
            offset + (self.number_values - 1) * increment,
            self.number_values)

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
        return (np.datetime64(start_time) +
                (relative_time * unit_correction).astype(time_type))

    def _initialise_data(self, memmap_dir=None):
        """Initialise data array to zeros"""

        if self.number_values == 0:
            pass
        elif self.data_type == types.DaqMxRawData:
            segment_obj = self._previous_segment_object
            self._scaler_insert_positions = {}
            for scaler in segment_obj.daqmx_metadata.scalers:
                self._scaler_data[scaler.scale_id] = self._new_numpy_array(
                    scaler.data_type.nptype, self.number_values, memmap_dir)
                self._scaler_insert_positions[scaler.scale_id] = 0
        elif self.data_type.nptype is None:
            self._data = []
        else:
            self._data = self._new_numpy_array(
                self.data_type.nptype, self.number_values, memmap_dir)
            self._data_insert_position = 0
            log.debug("Allocated %d sample slots for %s", len(self._data),
                      self.path)

    def _new_numpy_array(self, dtype, num_values, memmap_dir):
        """Initialise numpy array for data
        """
        if memmap_dir:
            memmap_file = tempfile.NamedTemporaryFile(
                mode='w+b', prefix="nptdms_", dir=memmap_dir)
            return np.memmap(
                memmap_file.file,
                mode='w+',
                shape=(num_values,),
                dtype=dtype)
        else:
            return np.zeros(num_values, dtype=dtype)

    def _update_data(self, new_data):
        """Update the object data with a new array of data"""

        log.debug("Adding %d data points to data for %s",
                  len(new_data), self.path)
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

    def _update_data_for_scaler(self, scale_id, new_data):
        """Append new DAQmx scaler data read from a segment
        """

        log.debug("Adding %d data points for object %s, scaler %d",
                  len(new_data), self.path, scale_id)
        data_array = self._scaler_data[scale_id]
        start_pos = self._scaler_insert_positions[scale_id]
        end_pos = start_pos + len(new_data)
        data_array[start_pos:end_pos] = new_data
        self._scaler_insert_positions[scale_id] += len(new_data)

    def as_dataframe(self, absolute_time=False, scaled_data=True):
        """
        Converts the TDMS object to a DataFrame

        :param absolute_time: Whether times should be absolute rather than
            relative to the start time.
        :return: The TDMS object data.
        :rtype: pandas.DataFrame
        """

        import pandas as pd

        def get_data(chan):
            if scaled_data:
                return chan.data
            else:
                return chan.raw_data

        # When absolute_time is True,
        # use the wf_start_time as offset for the time_track()
        try:
            time = self.time_track(absolute_time)
        except KeyError:
            time = None
        if self.channel is None:
            return pd.DataFrame.from_dict(OrderedDict(
                (ch.channel, pd.Series(get_data(ch)))
                for ch in self.tdms_file.group_channels(self.group)))
        else:
            return pd.DataFrame(
                get_data(self), index=time, columns=[self.path])

    @_property_builtin
    def data(self):
        """
        NumPy array containing data if there is data for this object,
        otherwise None.
        """
        if self._data is None and not self._scaler_data:
            # self._data is None if data segment is empty
            return np.empty((0, 1))
        if self._data_scaled is None:
            scale = scaling.get_scaling(self)
            if scale is None:
                self._data_scaled = self._data
            elif self._scaler_data:
                self._data_scaled = scale.scale_daqmx(self._scaler_data)
            else:
                self._data_scaled = scale.scale(self._data)

        return self._data_scaled

    @_property_builtin
    def raw_data(self):
        """
        The raw, unscaled data array.
        For unscaled objects this is the same as the data property.
        """
        if self._scaler_data:
            if len(self._scaler_data) == 1:
                return next(v for v in self._scaler_data.values())
            else:
                raise Exception(
                    "This object has data for multiple DAQmx scalers, "
                    "use the raw_scaler_data method to get raw data "
                    "for a scale_id")
        if self._data is None:
            # self._data is None if data segment is empty
            return np.empty((0, 1))
        return self._data

    def raw_scaler_data(self, scale_id):
        """
        Raw DAQmx scaler data for the given scale id
        """
        return self._scaler_data[scale_id]


class TdmsSegmentObject(object):
    """
    Describes an object in an individual TDMS file segment
    """

    __slots__ = [
        'tdms_object', 'number_values', 'data_size',
        'has_data', 'data_type', 'dimension', 'endianness',
        'daqmx_metadata']

    def __init__(self, tdms_object, endianness):
        self.tdms_object = tdms_object
        self.endianness = endianness

        self.number_values = 0
        self.data_size = 0
        self.has_data = True
        self.data_type = None
        self.dimension = 1

    def _read_metadata_mx(self, f):

        # Read the data type
        data_type_val = types.Uint32.read(f, self.endianness)
        try:
            self.data_type = types.tds_data_types[data_type_val]
        except KeyError:
            raise KeyError("Unrecognised data type: %s" % data_type_val)

        if self.tdms_object.data_type is not None \
           and self.data_type != self.tdms_object.data_type:

            raise ValueError(
                "Segment object doesn't have the same data "
                "type as previous segments.")
        else:
            self.tdms_object.data_type = self.data_type

        log.debug("DAQmx object data type: %r", self.tdms_object.data_type)

        info = DaqMxMetadata(f, self.endianness)
        log.debug("DAQmx metadata: %r", info)

        return info

    def _read_metadata(self, f):
        """Read object metadata and update object information"""

        raw_data_index = types.Uint32.read(f, self.endianness)

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
            self.data_size = info.chunk_size * sum(info.raw_data_widths)
            self.number_values = info.chunk_size
            # segment reading code relies on a single consistent raw
            # data width so assert that there is only one.
            assert(len(info.raw_data_widths) == 1)
            self.daqmx_metadata = info
            # fall through and read properties
        else:
            # Assume metadata format is legacy TDMS format.
            # raw_data_index gives the length of the index information.
            self.has_data = True
            self.tdms_object.has_data = True

            # Read the data type
            try:
                self.data_type = types.tds_data_types[
                    types.Uint32.read(f, self.endianness)]
            except KeyError:
                raise KeyError("Unrecognised data type")
            if (self.tdms_object.data_type is not None and
                    self.data_type != self.tdms_object.data_type):
                raise ValueError(
                    "Segment object doesn't have the same data "
                    "type as previous segments.")
            else:
                self.tdms_object.data_type = self.data_type
            log.debug("Object data type: %r", self.tdms_object.data_type)

            if (self.tdms_object.data_type.size is None and
                    self.tdms_object.data_type != types.String):
                raise ValueError(
                    "Unsupported data type: %r" % self.tdms_object.data_type)

            # Read data dimension
            self.dimension = types.Uint32.read(f, self.endianness)
            # In TDMS version 2.0, 1 is the only valid value for dimension
            if self.dimension != 1:
                log.warning("Data dimension is not 1")

            # Read number of values
            self.number_values = types.Uint64.read(f, self.endianness)

            # Variable length data types have total size
            if self.data_type in (types.String, ):
                self.data_size = types.Uint64.read(f, self.endianness)
            else:
                self.data_size = (
                    self.number_values *
                    self.data_type.size * self.dimension)

            log.debug(
                "Object number of values in segment: %d", self.number_values)

        # Read data properties
        num_properties = types.Uint32.read(f, self.endianness)
        log.debug("Reading %d properties", num_properties)
        for i in range(num_properties):
            prop_name, value = read_property(f, self.endianness)
            self.tdms_object.properties[prop_name] = value

    @property
    def path(self):
        return self.tdms_object.path

    @property
    def total_raw_data_width(self):
        if self.data_type == types.DaqMxRawData:
            return sum(self.daqmx_metadata.raw_data_widths)
        else:
            return self.data_type.size

    def _read_value(self, file):
        """Read a single value from the given file"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=1)
        return self.data_type.read(file, self.endianness)

    def _read_values(self, file, number_values):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=number_values)
        elif self.data_type == types.String:
            return read_string_data(file, number_values, self.endianness)
        data = self._new_segment_data()
        for i in range(number_values):
            data[i] = self.data_type.read(file, self.endianness)
        return data

    def _new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values


def read_string_data(file, number_values, endianness):
    """ Read string raw data

        This is stored as an array of offsets
        followed by the contiguous string data.
    """
    offsets = [0]
    for i in range(number_values):
        offsets.append(types.Uint32.read(file, endianness))
    strings = []
    for i in range(number_values):
        s = file.read(offsets[i + 1] - offsets[i])
        strings.append(s.decode('utf-8'))
    return strings


def read_property(f, endianness="<"):
    """ Read a property from a segment's metadata """

    prop_name = types.String.read(f, endianness)
    prop_data_type = types.tds_data_types[types.Uint32.read(f, endianness)]
    value = prop_data_type.read(f, endianness)
    log.debug("Property %s: %r", prop_name, value)
    return prop_name, value


def fromfile(file, dtype, count, *args, **kwargs):
    """Wrapper around np.fromfile to support any file-like object"""

    try:
        return np.fromfile(file, dtype=dtype, count=count, *args, **kwargs)
    except (TypeError, IOError, UnsupportedOperation):
        return np.frombuffer(
            file.read(count * np.dtype(dtype).itemsize),
            dtype=dtype, count=count, *args, **kwargs)


def read_interleaved_segment_bytes(f, bytes_per_row, num_values):
    """ Read a segment of interleaved data as rows of bytes
    """
    number_bytes = bytes_per_row * num_values
    combined_data = fromfile(f, dtype=np.uint8, count=number_bytes)
    try:
        # Reshape, so that one row is all bytes for all objects
        combined_data = combined_data.reshape(-1, bytes_per_row)
    except ValueError:
        # Probably incomplete segment at the end => try to clip data
        crop_len = (combined_data.shape[0] // bytes_per_row)
        crop_len *= bytes_per_row
        log.warning("Cropping data from %d to %d bytes to match segment "
                    "size derived from channels",
                    combined_data.shape[0], crop_len)
        combined_data = combined_data[:crop_len].reshape(-1, bytes_per_row)
    return combined_data
