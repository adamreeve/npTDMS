from copy import copy
from io import UnsupportedOperation
import os
import struct
import numpy as np

from nptdms import types
from nptdms.common import toc_properties
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)
_struct_unpack = struct.unpack

RAW_DATA_INDEX_NO_DATA = 0xFFFFFFFF
RAW_DATA_INDEX_MATCHES_PREVIOUS = 0x00000000


class BaseSegment(object):
    """ Abstract base class for TDMS segments
    """

    __slots__ = [
        'position', 'num_chunks', 'ordered_objects', 'toc_mask',
        'next_segment_offset', 'next_segment_pos',
        'raw_data_offset', 'data_position', 'final_chunk_proportion',
        'endianness', 'object_properties']

    def __init__(
            self, position, toc_mask, endianness, next_segment_offset,
            next_segment_pos, raw_data_offset, data_position):
        self.position = position
        self.toc_mask = toc_mask
        self.endianness = endianness
        self.next_segment_offset = next_segment_offset
        self.next_segment_pos = next_segment_pos
        self.raw_data_offset = raw_data_offset
        self.data_position = data_position
        self.num_chunks = 0
        self.final_chunk_proportion = 1.0
        self.ordered_objects = []
        self.object_properties = None

    def __repr__(self):
        return "<TdmsSegment at position %d>" % self.position

    def read_segment_objects(self, file, previous_segment_objects, previous_segment=None):
        """Read segment metadata section and update object information

        :param file: Open TDMS file
        :param previous_segment_objects: Dictionary of path to the most
            recently read segment object for a TDMS object.
        :param previous_segment: Previous segment in the file.
        """

        if not self.toc_mask & toc_properties['kTocMetaData']:
            self._reuse_previous_segment_metadata(previous_segment)
            return

        endianness = self.endianness

        new_obj_list = self.toc_mask & toc_properties['kTocNewObjList']
        if not new_obj_list:
            # In this case, there can be a list of new objects that
            # are appended, or previous objects can also be repeated
            # if their properties change.
            # Copy the list of objects for now, but any objects that have
            # metadata changed will need to be copied before being modified.
            self.ordered_objects = [
                o for o in previous_segment.ordered_objects]
            existing_objects = {o.path: (i, o) for (i, o) in enumerate(self.ordered_objects)}
        else:
            existing_objects = None

        log.debug("Reading segment object metadata at %d", file.tell())

        # First four bytes have number of objects in metadata
        num_objects_bytes = file.read(4)
        num_objects = _struct_unpack(endianness + 'L', num_objects_bytes)[0]

        for _ in range(num_objects):
            # Read the object path
            object_path = types.String.read(file, endianness)
            raw_data_index_header_bytes = file.read(4)
            raw_data_index_header = _struct_unpack(endianness + 'L', raw_data_index_header_bytes)[0]
            log.debug("Reading metadata for object %s with index header 0x%08x", object_path, raw_data_index_header)

            # Check whether we already have this object in our list from
            # the last segment
            (existing_object_index, existing_object) = (
                self._get_existing_object(existing_objects, object_path)
                if existing_objects is not None
                else (None, None))
            if existing_object_index is not None:
                self._update_existing_object(
                    object_path, existing_object_index, existing_object, raw_data_index_header, file)
            elif object_path in previous_segment_objects:
                previous_segment_obj = previous_segment_objects[object_path]
                self._reuse_previous_object(
                    object_path, previous_segment_obj, raw_data_index_header, file)
            else:
                segment_obj = self._new_segment_object(object_path)
                self.ordered_objects.append(segment_obj)
                if raw_data_index_header == RAW_DATA_INDEX_MATCHES_PREVIOUS:
                    raise ValueError("Raw data index for %s says to reuse previous structure, "
                                     "but we have not seen this object before" % object_path)
                elif raw_data_index_header != RAW_DATA_INDEX_NO_DATA:
                    segment_obj.has_data = True
                    segment_obj.read_raw_data_index(file, raw_data_index_header)

            self._read_object_properties(file, object_path)
        self._calculate_chunks()

    def _update_existing_object(
            self, object_path, existing_object_index, existing_object, raw_data_index_header, file):
        """ Update raw data index information for an object already in the list of segment objects
        """
        if raw_data_index_header == RAW_DATA_INDEX_NO_DATA:
            # Re-use object but leave data index information as set previously
            if existing_object.has_data:
                new_obj = copy(existing_object)
                new_obj.has_data = False
                self.ordered_objects[existing_object_index] = new_obj
        elif raw_data_index_header == RAW_DATA_INDEX_MATCHES_PREVIOUS:
            # Re-use object and ensure we set has data to true for this segment
            if not existing_object.has_data:
                new_obj = copy(existing_object)
                new_obj.has_data = True
                self.ordered_objects[existing_object_index] = new_obj
        else:
            # New segment metadata, or updates to existing data
            segment_obj = self._new_segment_object(object_path)
            segment_obj.has_data = True
            segment_obj.read_raw_data_index(file, raw_data_index_header)
            self.ordered_objects[existing_object_index] = segment_obj

    def _reuse_previous_object(
            self, object_path, previous_segment_obj, raw_data_index_header, file):
        """ Attempt to reuse raw data index information from a previous segment
        """
        if raw_data_index_header == RAW_DATA_INDEX_NO_DATA:
            # Re-use object but leave data index information as set previously
            if previous_segment_obj.has_data:
                segment_obj = copy(previous_segment_obj)
                segment_obj.has_data = False
            else:
                segment_obj = previous_segment_obj
        elif raw_data_index_header == RAW_DATA_INDEX_MATCHES_PREVIOUS:
            # Re-use previous object and ensure we set has data to true for this segment
            if not previous_segment_obj.has_data:
                segment_obj = copy(previous_segment_obj)
                segment_obj.has_data = True
            else:
                segment_obj = previous_segment_obj
        else:
            # Changed metadata in this segment
            segment_obj = self._new_segment_object(object_path)
            segment_obj.has_data = True
            segment_obj.read_raw_data_index(file, raw_data_index_header)
        self.ordered_objects.append(segment_obj)

    def _reuse_previous_segment_metadata(self, previous_segment):
        try:
            self.ordered_objects = previous_segment.ordered_objects
            self._calculate_chunks()
        except AttributeError:
            raise ValueError(
                "kTocMetaData is not set for segment but "
                "there is no previous segment")

    def _get_existing_object(self, existing_objects, object_path):
        """ Find an object already in the list of objects that are reused from the previous segment
        """
        try:
            return existing_objects[object_path]
        except KeyError:
            return None, None

    def _read_object_properties(self, file, object_path):
        """Read properties for an object in the segment
        """
        num_properties_bytes = file.read(4)
        num_properties = _struct_unpack(self.endianness + 'L', num_properties_bytes)[0]
        if num_properties > 0:
            log.debug("Reading %d properties", num_properties)
            if self.object_properties is None:
                self.object_properties = {}
            self.object_properties[object_path] = [
                read_property(file, self.endianness)
                for _ in range(num_properties)]

    def read_raw_data(self, f):
        """Read raw data from a TDMS segment

        :returns: A generator of RawDataChunk objects with raw channel data for
            objects in this segment.
        """

        if not self.toc_mask & toc_properties['kTocRawData']:
            yield RawDataChunk.empty()

        f.seek(self.data_position)

        total_data_size = self.next_segment_offset - self.raw_data_offset
        log.debug(
            "Reading %d bytes of data at %d in %d chunks",
            total_data_size, f.tell(), self.num_chunks)

        data_objects = [o for o in self.ordered_objects if o.has_data]
        for chunk in self._read_data_chunks(f, data_objects, self.num_chunks):
            yield chunk

    def read_raw_data_for_channel(self, f, channel_path, chunk_offset=0, num_chunks=None):
        """Read raw data from a TDMS segment

        :param f: Open TDMS file object
        :param channel_path: Path of channel to read data for
        :param chunk_offset: Index of chunk to begin reading from
        :param num_chunks: Number of chunks to read, or None to read to the end
        :returns: A generator of RawChannelDataChunk objects with raw channel data for
            a single channel in this segment.
        """

        if not self.toc_mask & toc_properties['kTocRawData']:
            yield RawChannelDataChunk.empty()

        f.seek(self.data_position)

        data_objects = [o for o in self.ordered_objects if o.has_data]
        chunk_size = self._get_chunk_size()

        if chunk_offset > 0:
            f.seek(chunk_size * chunk_offset, os.SEEK_CUR)
        stop_chunk = self.num_chunks if num_chunks is None else num_chunks + chunk_offset
        for chunk in self._read_channel_data_chunks(f, data_objects, channel_path, chunk_offset, stop_chunk):
            yield chunk

    def _calculate_chunks(self):
        """
        Work out the number of chunks the data is in, for cases
        where the meta data doesn't change at all so there is no
        lead in.
        """

        data_size = self._get_chunk_size()

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

    def _get_chunk_size(self):
        return sum([
            o.data_size
            for o in self.ordered_objects if o.has_data])

    def _read_data_chunks(self, file, data_objects, num_chunks):
        """ Read multiple data chunks at once
            In the base case we read each chunk individually but subclasses can override this
        """
        for chunk in range(num_chunks):
            yield self._read_data_chunk(file, data_objects, chunk)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        """ Read data from a chunk for all channels
        """
        raise NotImplementedError("Data chunk reading must be implemented in base classes")

    def _read_channel_data_chunks(self, file, data_objects, channel_path, chunk_offset, stop_chunk):
        """ Read multiple data chunks for a single channel at once
            In the base case we read each chunk individually but subclasses can override this
        """
        for chunk_index in range(chunk_offset, stop_chunk):
            yield self._read_channel_data_chunk(file, data_objects, chunk_index, channel_path)

    def _read_channel_data_chunk(self, file, data_objects, chunk_index, channel_path):
        """ Read data from a chunk for a single channel
        """
        # In the base case we can read data for all channels
        # and then select only the requested channel.
        # Derived classes can implement more optimised reading.
        data_chunk = self._read_data_chunk(file, data_objects, chunk_index)
        return BaseSegment._data_chunk_to_channel_chunk(data_chunk, channel_path)

    @staticmethod
    def _data_chunk_to_channel_chunk(data_chunk, channel_path):
        try:
            return data_chunk.channel_data[channel_path]
        except KeyError:
            return RawChannelDataChunk.empty()

    def _new_segment_object(self, object_path):
        """ Create a new segment object for a segment

        :param object_path: Path for the object
        """

        raise NotImplementedError("New segment object creation must be implemented in base classes")


class BaseSegmentObject(object):
    """ Abstract base class for an object in a TDMS segment
    """

    __slots__ = [
        'path', 'number_values', 'data_size',
        'has_data', 'data_type', 'endianness']

    def __init__(self, path, endianness):
        self.path = path
        self.number_values = 0
        self.data_size = 0
        self.has_data = False
        self.data_type = None
        self.endianness = endianness

    def read_raw_data_index(self, file, raw_data_index_header):
        """ Read the raw data index for a single object in a segment
        """
        raise NotImplementedError("Segment metadata reading must be implemented in base classes")

    @property
    def scaler_data_types(self):
        return None


class RawDataChunk(object):
    """Data read from a single chunk in a TDMS segment

    :ivar channel_data: A dictionary of channel data chunks.
        Keys are object paths and values are RawChannelDataChunk instances.
    """

    def __init__(self, channel_data):
        self.channel_data = channel_data

    @staticmethod
    def empty():
        return RawDataChunk({})

    @staticmethod
    def channel_data(data):
        channel_chunks = {
            path: RawChannelDataChunk.channel_data(d)
            for (path, d) in data.items()
        }
        return RawDataChunk(channel_chunks)

    @staticmethod
    def scaler_data(data):
        channel_chunks = {
            path: RawChannelDataChunk.scaler_data(d)
            for (path, d) in data.items()
        }
        return RawDataChunk(channel_chunks)


class RawChannelDataChunk(object):
    """Data read for a single channel from a single chunk in a TDMS segment

    :ivar data: Raw data in this chunk for a standard TDMS channel.
    :ivar scaler_data: A dictionary of scaler data in this segment for
        DAQmx raw data. Keys are the scaler id and values are data arrays.
    """

    def __init__(self, data, scaler_data):
        self.data = data
        self.scaler_data = scaler_data

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        elif self.scaler_data is not None:
            return next(len(d) for d in self.scaler_data.values())
        return 0

    @staticmethod
    def empty():
        return RawChannelDataChunk(None, None)

    @staticmethod
    def channel_data(data):
        return RawChannelDataChunk(data, None)

    @staticmethod
    def scaler_data(data):
        return RawChannelDataChunk(None, data)


def read_property(f, endianness="<"):
    """ Read a property from a segment's metadata """

    prop_name = types.String.read(f, endianness)
    prop_data_type = types.tds_data_types[types.Uint32.read(f, endianness)]
    value = prop_data_type.read(f, endianness)
    log.debug("Property '%s' = %r", prop_name, value)
    return prop_name, value


def fromfile(file, dtype, count, *args, **kwargs):
    """Wrapper around np.fromfile to support any file-like object"""

    try:
        return np.fromfile(file, dtype=dtype, count=count, *args, **kwargs)
    except (TypeError, IOError, UnsupportedOperation):
        return np.frombuffer(
            file.read(int(count * np.dtype(dtype).itemsize)),
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
