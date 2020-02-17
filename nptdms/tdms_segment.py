from copy import copy
from collections import defaultdict
from io import UnsupportedOperation
import os
import numpy as np

from nptdms import types
from nptdms.common import toc_properties
from nptdms.daqmx import DaqMxMetadata
from nptdms.log import log_manager
from nptdms.utils import OrderedDict


log = log_manager.get_logger(__name__)


class TdmsSegment(object):

    __slots__ = [
        'position', 'num_chunks', 'ordered_objects', 'toc', 'version',
        'next_segment_offset', 'next_segment_pos',
        'raw_data_offset', 'data_position', 'final_chunk_proportion',
        'endianness']

    def __init__(self, f):
        """Read the lead in section of a segment"""

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

    def read_metadata(
            self, f, previous_segment_objects, previous_segment=None):
        """Read segment metadata section and update object information

        :param f: Open TDMS file.
        :param previous_segment_objects: Dictionary of path to the most
            recently read segment object for a TDMS object.
        :param previous_segment: Previous segment in the file.
        """

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

        for _ in range(num_objects):
            # Read the object path
            object_path = types.String.read(f, self.endianness)

            # Add this segment object to the list of segment objects,
            # re-using any properties from previous segments.
            updating_existing = False
            if not self.toc["kTocNewObjList"]:
                # Search for the same object from the previous segment
                # object list.
                for obj in self.ordered_objects:
                    if obj.path == object_path:
                        updating_existing = True
                        log.debug("Updating object in segment list")
                        segment_obj = obj
                        break
            if not updating_existing:
                try:
                    prev_segment_obj = previous_segment_objects[object_path]
                    log.debug("Copying previous segment object for %s",
                              object_path)
                    segment_obj = copy(prev_segment_obj)
                except KeyError:
                    log.debug("Creating a new segment object for %s",
                              object_path)
                    segment_obj = TdmsSegmentObject(
                        object_path, self.endianness)
                self.ordered_objects.append(segment_obj)
            # Read the metadata for this object, updating any
            # data structure information and properties.
            segment_obj.read_metadata(f)

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
            # For DAQmxRawData, each channel in a segment has the same number
            # of values and contains the same raw data widths, so use
            # the first valid channel metadata to calcualte the data size.
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
        else:
            log.warning(
                "Data size %d is not a multiple of the "
                "chunk size %d. Will attempt to read last chunk",
                total_data_size, data_size)
            self.num_chunks = 1 + int(total_data_size // data_size)
            self.final_chunk_proportion = (
                float(chunk_remainder) / float(data_size))

    def read_raw_data(self, f):
        """Read raw data from a TDMS segment

        :returns: A generator of DataChunk objects with raw channel data for
            objects in this segment.
        """

        if not self.toc["kTocRawData"]:
            yield DataChunk.empty()

        f.seek(self.data_position)

        total_data_size = self.next_segment_offset - self.raw_data_offset
        log.debug(
            "Reading %d bytes of data at %d in %d chunks",
            total_data_size, f.tell(), self.num_chunks)

        for chunk in range(self.num_chunks):
            if self.toc['kTocDAQmxRawData']:
                data_objects = [o for o in self.ordered_objects if o.has_data]
                yield self._read_interleaved_daqmx(f, data_objects)
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
                    yield self._read_interleaved_numpy(f, data_objects)
                else:
                    yield self._read_interleaved(f, data_objects)
            else:
                yield self._read_contiguous_data(f, data_objects, chunk)

    def _read_contiguous_data(self, f, data_objects, chunk):
        """ Read contiguous (non-interleaved) data from a segment
        """
        log.debug("Data is contiguous")
        object_data = {}
        for obj in data_objects:
            if (chunk == (self.num_chunks - 1) and
                    self.final_chunk_proportion != 1.0):
                number_values = int(
                    obj.number_values *
                    self.final_chunk_proportion)
            else:
                number_values = obj.number_values
            object_data[obj.path] = obj.read_values(f, number_values)
        return DataChunk.channel_data(object_data)

    def _read_interleaved_daqmx(self, f, data_objects):
        """Read data from DAQmx data segment"""

        log.debug("Reading DAQmx data segment")

        # If we have DAQmx data, we expect all objects to have matching
        # raw data widths:
        all_daqmx = all(
            o.data_type == types.DaqMxRawData for o in data_objects)
        if not all_daqmx:
            raise Exception("Cannot read a mix of DAQmx and "
                            "non-DAQmx interleaved data")

        raw_data_widths = data_objects[0].daqmx_metadata.raw_data_widths
        chunk_size = data_objects[0].number_values
        scaler_data = defaultdict(dict)

        # Data for each set of raw data (corresponding to one card) is
        # interleaved separately, so read one after another
        for (raw_buffer_index, raw_data_width) in enumerate(raw_data_widths):
            # Read all data into 1 byte unsigned ints first
            combined_data = read_interleaved_segment_bytes(
                f, raw_data_width, chunk_size)

            # Now set arrays for each scaler of each channel where the scaler
            # data comes from this set of raw data
            for (i, obj) in enumerate(data_objects):
                scalers_for_raw_buffer_index = [
                    scaler for scaler in obj.daqmx_metadata.scalers
                    if scaler.raw_buffer_index == raw_buffer_index]
                for scaler in scalers_for_raw_buffer_index:
                    offset = scaler.raw_byte_offset
                    scaler_size = scaler.data_type.size
                    byte_columns = tuple(
                        range(offset, offset + scaler_size))
                    log.debug("Byte columns for channel %d scaler %d: %s",
                              i, scaler.scale_id, byte_columns)
                    # Select columns for this scaler, so that number of values
                    # will be number of bytes per point * number of data
                    # points. Then use ravel to flatten the results into a
                    # vector.
                    scaler_data = combined_data[:, byte_columns].ravel()
                    # Now set correct data type, so that the array length
                    # should be correct
                    scaler_data.dtype = (
                        scaler.data_type.nptype.newbyteorder(self.endianness))
                    scaler_data[obj.path][scaler.scale_id] = scaler_data

        return DataChunk.scaler_data(scaler_data)

    def _read_interleaved_numpy(self, f, data_objects):
        """Read interleaved data where all channels have a numpy type"""

        log.debug("Reading interleaved data all at once")

        # For non-DAQmx, simply use the data type sizes
        all_channel_bytes = sum(o.data_type.size for o in data_objects)
        log.debug("all_channel_bytes: %d", all_channel_bytes)

        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            f, all_channel_bytes, data_objects[0].number_values)

        # Now get arrays for each channel
        channel_data = {}
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
            channel_data[obj.path] = object_data
            data_pos += obj.data_type.size

        return DataChunk.channel_data(channel_data)

    def _read_interleaved(self, f, data_objects):
        """Read interleaved data that doesn't have a numpy type"""

        log.debug("Reading interleaved data point by point")
        object_data = {}
        points_added = {}
        for obj in data_objects:
            object_data[obj.path] = obj.new_segment_data()
            points_added[obj.path] = 0
        while any([points_added[o.path] < o.number_values
                   for o in data_objects]):
            for obj in data_objects:
                if points_added[obj.path] < obj.number_values:
                    object_data[obj.path][points_added[obj.path]] = (
                        obj.read_value(f))
                    points_added[obj.path] += 1

        return DataChunk.channel_data(object_data)


class TdmsSegmentObject(object):
    """
    Describes an object in an individual TDMS file segment
    """

    __slots__ = [
        'path', 'number_values', 'data_size',
        'has_data', 'data_type', 'dimension', 'endianness',
        'daqmx_metadata', 'properties']

    def __init__(self, path, endianness):
        self.path = path
        self.endianness = endianness

        self.number_values = 0
        self.data_size = 0
        self.has_data = True
        self.data_type = None
        self.dimension = 1
        self.daqmx_metadata = None
        self.properties = None

    def _read_metadata_mx(self, f):

        # Read the data type
        data_type_val = types.Uint32.read(f, self.endianness)
        try:
            self.data_type = types.tds_data_types[data_type_val]
        except KeyError:
            raise KeyError("Unrecognised data type: %s" % data_type_val)

        log.debug("DAQmx object data type: %r", self.data_type)

        info = DaqMxMetadata(f, self.endianness)
        log.debug("DAQmx metadata: %r", info)

        return info

    def read_metadata(self, f):
        """Read object metadata and update object information"""

        raw_data_index = types.Uint32.read(f, self.endianness)

        log.debug("Reading metadata for object %s", self.path)

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
        elif raw_data_index in (0x00001269, 0x00001369):
            # This is a DAQmx raw data segment.
            #    0x00001269 for segment containing Format Changing scaler.
            #    0x00001369 for segment containing Digital Line scaler.
            if raw_data_index == 0x00001369:
                # special scaling for DAQ's digital input lines?
                log.warning("DAQmx with Digital Line scaler has not tested")

            # DAQmx raw data format metadata has its own class
            self.has_data = True

            info = self._read_metadata_mx(f)
            self.dimension = info.dimension
            self.data_type = info.data_type
            # DAQmx format has special chunking
            self.data_size = info.chunk_size * sum(info.raw_data_widths)
            self.number_values = info.chunk_size
            self.daqmx_metadata = info
            # fall through and read properties
        else:
            # Metadata format is standard (non-DAQmx) TDMS format.
            # raw_data_index gives the length of the index information.
            self.has_data = True

            # Read the data type
            try:
                self.data_type = types.tds_data_types[
                    types.Uint32.read(f, self.endianness)]
            except KeyError:
                raise KeyError("Unrecognised data type")
            log.debug("Object data type: %r", self.data_type)

            if (self.data_type.size is None and
                    self.data_type != types.String):
                raise ValueError(
                    "Unsupported data type: %r" % self.data_type)

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
        self.properties = dict(
            read_property(f, self.endianness)
            for _ in range(num_properties))

    @property
    def total_raw_data_width(self):
        if self.data_type == types.DaqMxRawData:
            return sum(self.daqmx_metadata.raw_data_widths)
        else:
            return self.data_type.size

    def read_value(self, file):
        """Read a single value from the given file"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=1)
        return self.data_type.read(file, self.endianness)

    def read_values(self, file, number_values):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(self.endianness)
            return fromfile(file, dtype=dtype, count=number_values)
        elif self.data_type == types.String:
            return read_string_data(file, number_values, self.endianness)
        data = self.new_segment_data()
        for i in range(number_values):
            data[i] = self.data_type.read(file, self.endianness)
        return data

    def new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values


class DataChunk(object):
    """Data read from a single chunk in a TDMS segment

    :ivar raw_data: A dictionary of object data in this chunk for standard
        TDMS channels. Keys are object paths and values are numpy arrays.
    :ivar daqmx_raw_data: A dictionary of data in this segment for
        DAQmx raw data. Keys are object paths and values are dictionaries of
        numpy arrays keyed by scaler id.
    """

    def __init__(self, data, daqmx_data):
        self.raw_data = data
        self.daqmx_raw_data = daqmx_data

    @staticmethod
    def empty():
        return DataChunk({}, {})

    @staticmethod
    def channel_data(data):
        return DataChunk(data, {})

    @staticmethod
    def scaler_data(data):
        return DataChunk({}, data)


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
