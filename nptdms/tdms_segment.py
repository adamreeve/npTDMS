from copy import copy
import os
import numpy as np
import struct

from nptdms import types
from nptdms.common import toc_properties
from nptdms.log import log_manager
from nptdms.base_segment import (
    BaseSegmentObject,
    BaseDataReader,
    RawChannelDataChunk,
    RawDataChunk,
    read_interleaved_segment_bytes,
    data_chunk_to_channel_chunk,
    fromfile)
from nptdms.daqmx import (
    FORMAT_CHANGING_SCALER,
    DIGITAL_LINE_SCALER,
    DaqmxSegmentObject,
    DaqmxDataReader,
    get_daqmx_chunk_size,
    get_daqmx_final_chunk_lengths)


_struct_unpack = struct.unpack
log = log_manager.get_logger(__name__)


RAW_DATA_INDEX_NO_DATA = 0xFFFFFFFF
RAW_DATA_INDEX_MATCHES_PREVIOUS = 0x00000000


class TdmsSegment(object):
    """ Represents a segment of data in a TDMS file
    """

    __slots__ = [
        'position',
        'num_chunks',
        'ordered_objects',
        'toc_mask',
        'next_segment_pos',
        'data_position',
        'final_chunk_lengths_override',
        'object_index',
    ]

    def __init__(self, position, toc_mask, next_segment_pos, data_position):
        self.position = position
        self.toc_mask = toc_mask
        self.next_segment_pos = next_segment_pos
        self.data_position = data_position
        self.num_chunks = 0
        self.final_chunk_lengths_override = None
        self.ordered_objects = None
        self.object_index = None

    def __repr__(self):
        return "<TdmsSegment at position %d>" % self.position

    def read_segment_objects(self, file, previous_segment_objects, index_cache, previous_segment, segment_incomplete):
        """Read segment metadata section and update object information

        :param file: Open TDMS file
        :param previous_segment_objects: Dictionary of path to the most
            recently read segment object for a TDMS object.
        :param index_cache: A SegmentIndexCache instance, or None if segment indexes are not required.
        :param previous_segment: Previous segment in the file.
        :param segment_incomplete: Whether the next segment offset was not set.
        """

        if not self.toc_mask & toc_properties['kTocMetaData']:
            self._reuse_previous_segment_metadata(previous_segment, segment_incomplete)
            return

        endianness = '>' if (self.toc_mask & toc_properties['kTocBigEndian']) else '<'

        new_obj_list = self.toc_mask & toc_properties['kTocNewObjList']
        if new_obj_list or not previous_segment:
            self.ordered_objects = []
            existing_objects = None
        else:
            # In this case, there can be a list of new objects that
            # are appended, or previous objects can also be repeated
            # if their properties change.
            # Copy the list of objects for now, but any objects that have
            # metadata changed will need to be copied before being modified.
            self.ordered_objects = previous_segment.ordered_objects[:]
            existing_objects = {o.path: (i, o) for (i, o) in enumerate(self.ordered_objects)}

        log.debug("Reading segment object metadata at %d", file.tell())

        # First four bytes have number of objects in metadata
        num_objects_bytes = file.read(4)
        num_objects = _struct_unpack(endianness + 'L', num_objects_bytes)[0]
        properties = None

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
                    existing_object_index, existing_object, raw_data_index_header, file, endianness)
            elif object_path in previous_segment_objects:
                previous_segment_obj = previous_segment_objects[object_path]
                self._reuse_previous_object(
                    previous_segment_obj, raw_data_index_header, file, endianness)
            else:
                segment_obj = self._new_segment_object(object_path, raw_data_index_header)
                self.ordered_objects.append(segment_obj)
                if raw_data_index_header == RAW_DATA_INDEX_MATCHES_PREVIOUS:
                    raise ValueError("Raw data index for %s says to reuse previous structure, "
                                     "but we have not seen this object before" % object_path)
                elif raw_data_index_header != RAW_DATA_INDEX_NO_DATA:
                    segment_obj.has_data = True
                    segment_obj.read_raw_data_index(file, raw_data_index_header, endianness)

            object_properties = self._read_object_properties(file, endianness)
            if object_properties is not None:
                if properties is None:
                    properties = {}
                properties[object_path] = object_properties

        if index_cache is not None:
            self.object_index = index_cache.get_index(self.ordered_objects)
        self._calculate_chunks(segment_incomplete)
        return properties

    def get_segment_object(self, object_path):
        try:
            obj_index = self.object_index[object_path]
            return self.ordered_objects[obj_index]
        except KeyError:
            return None

    def _update_existing_object(
            self, existing_object_index, existing_object, raw_data_index_header, file, endianness):
        """ Update raw data index information for an object already in the list of segment objects
        """
        object_path = existing_object.path
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
            segment_obj = self._new_segment_object(object_path, raw_data_index_header)
            segment_obj.has_data = True
            segment_obj.read_raw_data_index(file, raw_data_index_header, endianness)
            self.ordered_objects[existing_object_index] = segment_obj

    def _reuse_previous_object(
            self, previous_segment_obj, raw_data_index_header, file, endianness):
        """ Attempt to reuse raw data index information from a previous segment
        """
        object_path = previous_segment_obj.path
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
            segment_obj = self._new_segment_object(object_path, raw_data_index_header)
            segment_obj.has_data = True
            segment_obj.read_raw_data_index(file, raw_data_index_header, endianness)
        self.ordered_objects.append(segment_obj)

    def _reuse_previous_segment_metadata(self, previous_segment, segment_incomplete):
        try:
            self.ordered_objects = previous_segment.ordered_objects
            self.object_index = previous_segment.object_index
            self._calculate_chunks(segment_incomplete)
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

    def _read_object_properties(self, file, endianness):
        """Read properties for an object in the segment
        """
        num_properties_bytes = file.read(4)
        num_properties = _struct_unpack(endianness + 'L', num_properties_bytes)[0]
        if num_properties > 0:
            log.debug("Reading %d properties", num_properties)
            return [read_property(file, endianness) for _ in range(num_properties)]
        else:
            return None

    def read_raw_data(self, f):
        """Read raw data from a TDMS segment

        :returns: A generator of RawDataChunk objects with raw channel data for
            objects in this segment.
        """

        if not self.toc_mask & toc_properties['kTocRawData']:
            yield RawDataChunk.empty()

        f.seek(self.data_position)

        total_data_size = self.next_segment_pos - self.data_position
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

    def _calculate_chunks(self, segment_incomplete):
        """
        Work out the number of chunks the data is in, for cases
        where the meta data doesn't change at all so there is no
        lead in.
        """

        data_size = self._get_chunk_size()

        total_data_size = self.next_segment_pos - self.data_position
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
            self.final_chunk_lengths_override = self._compute_final_chunk_lengths(
                data_size, chunk_remainder, segment_incomplete)

    def _compute_final_chunk_lengths(self, chunk_size, chunk_remainder, segment_incomplete):
        """Compute object data lengths for a final chunk that has less data than expected
        """
        if self._have_daqmx_objects():
            return get_daqmx_final_chunk_lengths(self.ordered_objects, chunk_remainder)

        obj_chunk_sizes = {}

        if any(o for o in self.ordered_objects if o.has_data and o.data_type.size is None):
            # Don't try to handle truncated segments with unsized data
            return obj_chunk_sizes

        interleaved_data = self.toc_mask & toc_properties['kTocInterleavedData']
        if interleaved_data or not segment_incomplete:
            for obj in self.ordered_objects:
                if not obj.has_data:
                    continue
                obj_chunk_sizes[obj.path] = (obj.number_values * chunk_remainder) // chunk_size
        else:
            # Have contiguous truncated data
            for obj in self.ordered_objects:
                if not obj.has_data:
                    continue
                data_size = obj.number_values * obj.data_type.size
                if chunk_remainder > data_size:
                    obj_chunk_sizes[obj.path] = obj.number_values
                    chunk_remainder -= data_size
                else:
                    obj_chunk_sizes[obj.path] = chunk_remainder // obj.data_type.size
                    break

        return obj_chunk_sizes

    def _new_segment_object(self, object_path, raw_data_index_header):
        """ Create a new segment object for a segment

        :param object_path: Path for the object
        :param raw_data_index_header: Integer raw data index header value
        """
        if raw_data_index_header in (FORMAT_CHANGING_SCALER, DIGITAL_LINE_SCALER):
            return DaqmxSegmentObject(object_path)
        return TdmsSegmentObject(object_path)

    def _get_chunk_size(self):
        if self._have_daqmx_objects():
            return get_daqmx_chunk_size(self.ordered_objects)
        return sum(
            o.data_size
            for o in self.ordered_objects if o.has_data)

    def _read_data_chunks(self, file, data_objects, num_chunks):
        """ Read multiple data chunks at once
            In the base case we read each chunk individually but subclasses can override this
        """
        reader = self._get_data_reader()
        for chunk in reader.read_data_chunks(file, data_objects, num_chunks):
            yield chunk

    def _read_channel_data_chunks(self, file, data_objects, channel_path, chunk_offset, stop_chunk):
        """ Read multiple data chunks for a single channel at once
            In the base case we read each chunk individually but subclasses can override this
        """
        reader = self._get_data_reader()
        for chunk in reader.read_channel_data_chunks(file, data_objects, channel_path, chunk_offset, stop_chunk):
            yield chunk

    def _get_data_reader(self):
        endianness = '>' if (self.toc_mask & toc_properties['kTocBigEndian']) else '<'
        if self._have_daqmx_objects():
            return DaqmxDataReader(self.num_chunks, self.final_chunk_lengths_override, endianness)
        elif self._have_interleaved_data():
            return InterleavedDataReader(self.num_chunks, self.final_chunk_lengths_override, endianness)
        else:
            return ContiguousDataReader(self.num_chunks, self.final_chunk_lengths_override, endianness)

    def _have_daqmx_objects(self):
        data_obj_count = 0
        daqmx_count = 0
        for o in self.ordered_objects:
            if o.has_data:
                data_obj_count += 1
                if isinstance(o, DaqmxSegmentObject):
                    daqmx_count += 1
        if daqmx_count == 0:
            return False
        if daqmx_count == data_obj_count:
            return True
        if daqmx_count > 0:
            raise Exception("Cannot read mixed DAQmx and non-DAQmx data")
        return False

    def _have_interleaved_data(self):
        """ Whether data in this segment is interleaved. Assumes data is not DAQmx.
        """
        if not (self.toc_mask & toc_properties['kTocInterleavedData']):
            return False

        data_obj_count = 0
        unsized_type_count = 0
        for o in self.ordered_objects:
            if o.has_data:
                data_obj_count += 1
                if o.data_type.size is None:
                    unsized_type_count += 1
        if unsized_type_count == 0:
            return True
        elif unsized_type_count == 1 and data_obj_count == 1:
            # Some files may have segments with an interleaved data flag set but contain a single
            # channel of string data which must be read as a contiguous data chunk.
            return False
        else:
            raise ValueError("Cannot read interleaved segment containing channels with unsized types")


class InterleavedDataReader(BaseDataReader):
    """ Reads data in a TDMS segment with interleaved data
    """

    def read_data_chunks(self, file, data_objects, num_chunks):
        """ Read multiple data chunks at once
        """
        if len(data_objects) == 0:
            return []
        same_length = (len(
            set((o.number_values for o in data_objects))) == 1)
        if not same_length:
            raise ValueError("Cannot read interleaved data with different chunk sizes")
        return [self._read_interleaved_chunks(file, data_objects, num_chunks)]

    def read_channel_data_chunks(self, file, data_objects, channel_path, chunk_offset, stop_chunk):
        """ Read multiple data chunks for a single channel at once
        """
        num_chunks = stop_chunk - chunk_offset
        all_chunks = self.read_data_chunks(file, data_objects, num_chunks)
        return [data_chunk_to_channel_chunk(chunk, channel_path) for chunk in all_chunks]

    def _read_data_chunk(self, file, data_objects, chunk_index):
        """ Not used for interleaved data, multiple chunks are read at once
        """
        raise NotImplementedError("Reading a single chunk is not implemented for interleaved data")

    def _read_interleaved_chunks(self, file, data_objects, num_chunks):
        """Read interleaved data where all channels have a sized data type and the same length
        """
        total_data_width = sum(o.data_type.size for o in data_objects)
        log.debug("Reading interleaved data all at once. total_data_width: %d", total_data_width)

        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            file, total_data_width, data_objects[0].number_values * num_chunks)

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
            object_data = obj.data_type.from_bytes(object_data, self.endianness)
            channel_data[obj.path] = object_data
            data_pos += obj.data_type.size

        return RawDataChunk.channel_data(channel_data)


class ContiguousDataReader(BaseDataReader):
    """ Reads data in a TDMS segment with contiguous (non-interleaved) data
    """

    def _read_data_chunk(self, file, data_objects, chunk_index):
        log.debug("Reading contiguous data chunk")
        object_data = {}
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            object_data[obj.path] = obj.read_values(file, number_values, self.endianness)
        return RawDataChunk.channel_data(object_data)

    def _read_channel_data_chunk(self, file, data_objects, chunk_index, channel_path):
        """ Read data from a chunk for a single channel
        """
        channel_data = RawChannelDataChunk.empty()
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            if obj.path == channel_path:
                channel_data = RawChannelDataChunk.channel_data(obj.read_values(file, number_values, self.endianness))
            elif number_values == obj.number_values:
                # Seek over data for other channel data
                file.seek(obj.data_size, os.SEEK_CUR)
            else:
                # In last chunk with reduced chunk size
                if obj.data_type.size is None:
                    # Type is unsized (eg. string), try reading number of values
                    obj.read_values(file, number_values, self.endianness)
                else:
                    file.seek(obj.data_type.size * number_values, os.SEEK_CUR)
        return channel_data

    def _get_channel_number_values(self, obj, chunk_index):
        if chunk_index == (self.num_chunks - 1) and self.final_chunk_lengths_override is not None:
            return self.final_chunk_lengths_override.get(obj.path, 0)
        else:
            return obj.number_values


class TdmsSegmentObject(BaseSegmentObject):
    """ A standard (non DAQmx) TDMS segment object
    """

    __slots__ = []

    def read_raw_data_index(self, f, raw_data_index_header, endianness):
        # Metadata format is standard (non-DAQmx) TDMS format.
        # raw_data_index_header gives the length of the index information.

        # Read type, dimension and chunk size at once for performance
        index_bytes = f.read(16)
        (data_type, dimension, number_values) = _struct_unpack(endianness + 'LLQ', index_bytes)
        self.number_values = number_values

        # Get data type
        try:
            self.data_type = types.tds_data_types[data_type]
        except KeyError:
            raise KeyError("Unrecognised data type")

        log.debug(
            "Object data type: %s\nObject number of values in segment: %d",
            self.data_type.__name__, self.number_values)

        if (self.data_type.size is None and
                self.data_type != types.String):
            raise ValueError(
                "Unsupported data type: %r" % self.data_type)

        # In TDMS version 2.0, 1 is the only valid value for dimension
        if dimension != 1:
            raise ValueError("Data dimension is not 1")

        # Variable length data types have total size
        if self.data_type == types.String:
            self.data_size = types.Uint64.read(f, endianness)
        else:
            self.data_size = self.number_values * self.data_type.size

    def read_values(self, file, number_values, endianness):
        """Read all values for this object from a contiguous segment"""

        if self.data_type.nptype is not None:
            dtype = self.data_type.nptype.newbyteorder(endianness)
            return fromfile(file, dtype=dtype, count=number_values)
        elif self.data_type.size is not None:
            byte_data = fromfile(file, dtype=np.dtype('uint8'), count=number_values * self.data_type.size)
            return self.data_type.from_bytes(byte_data, endianness)
        else:
            return self.data_type.read_values(file, number_values, endianness)

    def new_segment_data(self):
        """Return a new array to read the data of the current section into"""

        if self.data_type.nptype is not None:
            return np.zeros(self.number_values, dtype=self.data_type.nptype)
        else:
            return [None] * self.number_values


class SegmentIndexCache(object):
    """ Caches dictionaries for indexing segment object lists using object paths
    """

    def __init__(self):
        self._indexes = {}

    def get_index(self, object_list):
        key = ObjectListKey(object_list)
        try:
            return self._indexes[key]
        except KeyError:
            index = dict((o.path, i) for (i, o) in enumerate(object_list))
            self._indexes[key] = index
            return index


class ObjectListKey(object):
    """ Wraps a list of objects for using as a cache key, where we are only concerned with the object paths
    """

    __slots__ = ['objects', '_hash']

    def __init__(self, objects):
        self.objects = objects
        hash_result = 0
        for obj in objects:
            # This is order independent, but it's unlikely that different segments
            # would have the same objects but in a different order.
            hash_result = hash_result ^ hash(obj.path)
        self._hash = hash_result

    def __eq__(self, other):
        return len(self.objects) == len(other.objects) and all(
            oa.path == ob.path for (oa, ob) in zip(self.objects, other.objects))

    def __hash__(self):
        return self._hash


def read_property(f, endianness="<"):
    """ Read a property from a segment's metadata """

    prop_name = types.String.read(f, endianness)
    prop_data_type = types.tds_data_types[types.Uint32.read(f, endianness)]
    value = prop_data_type.read(f, endianness)
    log.debug("Property '%s' = %r", prop_name, value)
    return prop_name, value
