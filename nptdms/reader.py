""" Lower level TDMS reader API that allows finer grained reading of data
"""

import logging
import os
import numpy as np

from nptdms import types
from nptdms.common import ObjectPath, toc_properties
from nptdms.utils import Timer, OrderedDict
from nptdms.base_segment import RawChannelDataChunk
from nptdms.tdms_segment import ContiguousDataSegment, InterleavedDataSegment
from nptdms.daqmx import DaqmxSegment
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class TdmsReader(object):
    """ Reads metadata and data from a TDMS file.

    :ivar object_metadata: Dictionary of object path to ObjectMetadata
    """

    def __init__(self, tdms_file):
        """ Initialise a new TdmsReader

        :param tdms_file: Either the path to the tdms file to read
            as a string or pathlib.Path, or an already opened file.
        """
        self._segments = None
        self._prev_segment_objects = {}
        self.object_metadata = OrderedDict()
        self._file_path = None
        self._index_file_path = None

        self._segment_channel_offsets = None
        self._segment_chunk_sizes = None

        if hasattr(tdms_file, "read"):
            # Is a file
            self._file = tdms_file
        else:
            # Is path to a file
            self._file_path = str(tdms_file)
            self._file = open(self._file_path, 'rb')
            index_file_path = self._file_path + '_index'
            if os.path.isfile(index_file_path):
                self._index_file_path = index_file_path

    def close(self):
        if self._file is None:
            # Already closed
            return

        if self._file_path is not None:
            # File path was provided so we opened the file and
            # should close it.
            self._file.close()
        # Otherwise always remove reference to the file
        self._file = None

    def read_metadata(self):
        """ Read all metadata and structure information from a TdmsFile
        """
        self._ensure_open()

        if self._index_file_path is not None:
            reading_index_file = True
            file = open(self._index_file_path, 'rb')
        else:
            reading_index_file = False
            file = self._file

        self._segments = []
        segment_position = 0
        try:
            with Timer(log, "Read metadata"):
                # Read metadata first to work out how much space we need
                previous_segment = None
                while True:
                    start_position = file.tell()
                    try:
                        segment = self._read_segment_metadata(
                            file, segment_position, previous_segment, reading_index_file)
                    except EOFError:
                        # We've finished reading the file
                        break

                    self._update_object_metadata(segment)
                    self._update_object_properties(segment)
                    self._segments.append(segment)
                    previous_segment = segment

                    segment_position = segment.next_segment_pos
                    if reading_index_file:
                        lead_size = 7 * 4
                        file.seek(start_position + lead_size + segment.raw_data_offset, os.SEEK_SET)
                    else:
                        file.seek(segment.next_segment_pos, os.SEEK_SET)
        finally:
            if reading_index_file:
                file.close()

    def read_raw_data(self):
        """ Read raw data from all segments, chunk by chunk

        :returns: A generator that yields RawDataChunk objects
        """
        self._ensure_open()
        if self._segments is None:
            raise RuntimeError(
                "Cannot read data unless metadata has first been read")
        for segment in self._segments:
            self._verify_segment_start(segment)
            for chunk in segment.read_raw_data(self._file):
                yield chunk

    def read_raw_data_for_channel(self, channel_path, offset=0, length=None):
        """ Read raw data for a single channel, chunk by chunk

        :param channel_path: The path of the channel object to read data for
        :param offset: Initial position to read data from.
        :param length: Number of values to attempt to read.
            If None, then all values starting from the offset will be read.
            Fewer values will be returned if attempting to read beyond the end of the available data.
        :returns: A generator that yields RawChannelDataChunk objects
        """
        self._ensure_open()
        if self._segments is None:
            raise RuntimeError("Cannot read data unless metadata has first been read")

        if self._segment_channel_offsets is None:
            with Timer(log, "Build data index"):
                self._build_index()
        segment_offsets = self._segment_channel_offsets[channel_path]
        chunk_sizes = self._segment_chunk_sizes[channel_path]

        object_metadata = self.object_metadata[channel_path]
        if length is None:
            length = object_metadata.num_values - offset
        end_index = offset + length

        # Binary search to find first and last segments to read
        start_segment = np.searchsorted(segment_offsets, offset, side='right')
        end_segment = np.searchsorted(segment_offsets, end_index, side='left')

        segment_index = start_segment
        for segment in self._segments[start_segment:end_segment + 1]:
            self._verify_segment_start(segment)
            # By default, read all chunks in a segment
            chunk_offset = 0
            num_chunks = segment.num_chunks
            chunk_size = chunk_sizes[segment_index]
            segment_start_index = 0 if segment_index == 0 else segment_offsets[segment_index - 1]
            remaining_values_to_skip = 0
            remaining_values_to_trim = 0

            # For the first and last segments, we may not need to read all chunks,
            # and may need to trim some data from the beginning or end of the chunk.
            if segment_index == start_segment:
                num_values_to_skip = offset - segment_start_index
                chunk_offset = num_values_to_skip // chunk_size
                remaining_values_to_skip = num_values_to_skip % chunk_size
                num_chunks -= chunk_offset
            if segment_index == end_segment:
                # Note: segment_index may be both start and end
                segment_end_index = segment_offsets[segment_index]
                num_values_to_trim = segment_end_index - end_index

                # Account for segments where the final chunk is truncated
                final_chunk_size = (segment_end_index - segment_start_index) % chunk_size
                final_chunk_size = chunk_size if final_chunk_size == 0 else final_chunk_size
                if num_values_to_trim >= final_chunk_size:
                    num_chunks -= 1
                    num_values_to_trim -= final_chunk_size

                num_chunks -= num_values_to_trim // chunk_size
                remaining_values_to_trim = num_values_to_trim % chunk_size

            for i, chunk in enumerate(
                    segment.read_raw_data_for_channel(self._file, channel_path, chunk_offset, num_chunks)):
                skip = remaining_values_to_skip if i == 0 else 0
                trim = remaining_values_to_trim if i == (num_chunks - 1) else 0
                yield _trim_channel_chunk(chunk, skip, trim)

            segment_index += 1

    def read_channel_chunk_for_index(self, channel_path, index):
        """ Read the chunk containing the given index

        :returns: Tuple of raw channel data chunk and the integer offset to the beginning of the chunk
        :rtype: (RawChannelDataChunk, int)
        """
        self._ensure_open()
        if self._segments is None:
            raise RuntimeError("Cannot read data unless metadata has first been read")

        if self._segment_channel_offsets is None:
            with Timer(log, "Build data index"):
                self._build_index()
        segment_offsets = self._segment_channel_offsets[channel_path]

        # Binary search to find the segment to read
        segment_index = np.searchsorted(segment_offsets, index, side='right')
        segment = self._segments[segment_index]
        chunk_size = self._segment_chunk_sizes[channel_path][segment_index]
        segment_start_index = segment_offsets[segment_index - 1] if segment_index > 0 else 0

        index_in_segment = index - segment_start_index
        chunk_index = index_in_segment // chunk_size

        self._verify_segment_start(segment)
        chunk_data = next(segment.read_raw_data_for_channel(self._file, channel_path, chunk_index, 1))
        chunk_offset = segment_start_index + chunk_index * chunk_size
        return chunk_data, chunk_offset

    def _read_segment_metadata(
            self, file, segment_position, previous_segment=None, is_index_file=False):
        (position, toc_mask, endianness, data_position, raw_data_offset,
         next_segment_offset, next_segment_pos) = self._read_lead_in(file, segment_position, is_index_file)

        segment_args = (
            position, toc_mask, endianness, next_segment_offset,
            next_segment_pos, raw_data_offset, data_position)
        if toc_mask & toc_properties['kTocDAQmxRawData']:
            segment = DaqmxSegment(*segment_args)
        elif toc_mask & toc_properties['kTocInterleavedData']:
            segment = InterleavedDataSegment(*segment_args)
        else:
            segment = ContiguousDataSegment(*segment_args)

        segment.read_segment_objects(
            file, self._prev_segment_objects, previous_segment)
        return segment

    def _read_lead_in(self, file, segment_position, is_index_file=False):
        expected_tag = b'TDSh' if is_index_file else b'TDSm'
        tag = file.read(4)
        if tag == b'':
            raise EOFError
        if tag != expected_tag:
            raise ValueError(
                "Segment does not start with %r, but with %r" % (expected_tag, tag))

        log.debug("Reading segment at %d", segment_position)

        # Next four bytes are table of contents mask
        toc_mask = types.Int32.read(file)

        if log.isEnabledFor(logging.DEBUG):
            for prop_name, prop_mask in toc_properties.items():
                prop_is_set = (toc_mask & prop_mask) != 0
                log.debug("Property %s is %s", prop_name, prop_is_set)

        endianness = '>' if (toc_mask & toc_properties['kTocBigEndian']) else '<'

        # Next four bytes are version number
        version = types.Int32.read(file, endianness)
        if version not in (4712, 4713):
            log.warning("Unrecognised version number.")

        # Now 8 bytes each for the offset values
        next_segment_offset = types.Uint64.read(file, endianness)
        raw_data_offset = types.Uint64.read(file, endianness)

        # Calculate data and next segment position
        lead_size = 7 * 4
        data_position = segment_position + lead_size + raw_data_offset
        if next_segment_offset == 0xFFFFFFFFFFFFFFFF:
            # Segment size is unknown. This can happen if LabVIEW crashes.
            # Try to read until the end of the file.
            log.warning(
                "Last segment of file has unknown size, "
                "will attempt to read to the end of the file")
            next_segment_pos = self._get_data_file_size()
            next_segment_offset = next_segment_pos - segment_position - lead_size
        else:
            log.debug("Next segment offset = %d, raw data offset = %d",
                      next_segment_offset, raw_data_offset)
            log.debug("Data size = %d b",
                      next_segment_offset - raw_data_offset)
            next_segment_pos = (
                    segment_position + next_segment_offset + lead_size)

        return (segment_position, toc_mask, endianness, data_position, raw_data_offset,
                next_segment_offset, next_segment_pos)

    def _verify_segment_start(self, segment):
        """ When reading data for a segment, check for the TDSm tag at the start of the segment in an attempt
            to detect any mismatch between tdms and tdms_index files.
        """
        position = segment.position
        self._file.seek(segment.position)
        expected_tag = b'TDSm'
        tag = self._file.read(4)
        if tag != expected_tag:
            raise ValueError(
                "Attempted to read data segment at position {0} but did not find segment start header. ".format(
                    position) +
                "Check that the tdms_index file matches the tdms data file.")

    def _get_data_file_size(self):
        current_pos = self._file.tell()
        self._file.seek(0, os.SEEK_END)
        end_pos = self._file.tell()
        self._file.seek(current_pos, os.SEEK_SET)
        return end_pos

    def _update_object_metadata(self, segment):
        """ Update object metadata using the metadata read from a single segment
        """
        for segment_object in segment.ordered_objects:
            path = segment_object.path
            self._prev_segment_objects[path] = segment_object

            object_metadata = self._get_or_create_object(path)
            object_metadata.num_values += _number_of_segment_values(segment_object, segment)
            _update_object_data_type(path, object_metadata, segment_object)
            _update_object_scaler_data_types(path, object_metadata, segment_object)

    def _update_object_properties(self, segment):
        """ Update object properties using any properties in a segment
        """
        if segment.object_properties is not None:
            for path, properties in segment.object_properties.items():
                object_metadata = self._get_or_create_object(path)
                for prop, val in properties:
                    object_metadata.properties[prop] = val

    def _get_or_create_object(self, path):
        """ Get existing object metadata or create metadata for a new object
        """
        try:
            return self.object_metadata[path]
        except KeyError:
            obj = ObjectMetadata()
            self.object_metadata[path] = obj
            return obj

    def _build_index(self):
        """ Builds an index into the segment data for faster lookup of values

            _segment_channel_offsets provides data offset at the end of each segment per channel
            _segment_chunk_sizes provides chunk sizes in each segment per channel
        """
        data_objects = [
            path
            for (path, obj) in self.object_metadata.items()
            if ObjectPath.from_string(path).is_channel]
        num_segments = len(self._segments)

        segment_num_values = {
            path: np.zeros(num_segments, dtype=np.int64) for path in data_objects}
        segment_chunk_sizes = {
            path: np.zeros(num_segments, dtype=np.int64) for path in data_objects}

        for i, segment in enumerate(self._segments):
            for obj in segment.ordered_objects:
                if not obj.has_data:
                    continue
                segment_chunk_sizes[obj.path][i] = obj.number_values if obj.has_data else 0
                segment_num_values[obj.path][i] = _number_of_segment_values(obj, segment)

        self._segment_chunk_sizes = segment_chunk_sizes
        self._segment_channel_offsets = {
            path: np.cumsum(segment_count) for (path, segment_count) in segment_num_values.items()}

    def _ensure_open(self):
        if self._file is None:
            raise RuntimeError(
                "Cannot read data after the underlying TDMS reader is closed")


def _number_of_segment_values(segment_object, segment):
    """ Compute the number of values an object has in a segment
    """
    if not segment_object.has_data:
        return 0
    num_chunks = segment.num_chunks
    final_chunk_proportion = segment.final_chunk_proportion
    if final_chunk_proportion == 1.0:
        return segment_object.number_values * num_chunks
    else:
        return (segment_object.number_values * (num_chunks - 1) +
                int(segment_object.number_values * final_chunk_proportion))


def _update_object_data_type(path, obj, segment_object):
    """ Update the data type for an object using its segment metadata
    """
    if obj.data_type is not None and obj.data_type != segment_object.data_type:
        raise ValueError(
            "Segment data doesn't have the same type as previous "
            "segments for objects %s. Expected type %s but got %s" %
            (path, obj.data_type, segment_object.data_type))
    obj.data_type = segment_object.data_type


def _update_object_scaler_data_types(path, obj, segment_object):
    """ Update the DAQmx scaler data types for an object using its segment metadata
    """
    if segment_object.scaler_data_types is not None:
        if obj.scaler_data_types is not None and obj.scaler_data_types != segment_object.scaler_data_types:
            raise ValueError(
                "Segment data doesn't have the same scaler data types as previous "
                "segments for objects %s. Expected types %s but got %s" %
                (path, obj.scaler_data_types, segment_object.scaler_data_types))
        obj.scaler_data_types = segment_object.scaler_data_types


class ObjectMetadata(object):
    """ Stores information about an object in a TDMS file
    """
    def __init__(self):
        self.properties = OrderedDict()
        self.data_type = None
        self.scaler_data_types = None
        self.num_values = 0


def _trim_channel_chunk(chunk, skip=0, trim=0):
    if skip == 0 and trim == 0:
        return chunk
    data = None
    scaler_data = None
    if chunk.data is not None:
        data = chunk.data[skip:len(chunk.data) - trim]
    if chunk.scaler_data is not None:
        scaler_data = {
            scale_id: d[skip:len(d) - trim]
            for (scale_id, d) in chunk.scaler_data.items()}
    return RawChannelDataChunk(data, scaler_data)
