import logging
import os
import numpy as np

from nptdms import types
from nptdms.common import toc_properties
from nptdms.base_segment import (
    BaseSegment,
    BaseSegmentObject,
    ChannelDataChunk,
    DataChunk,
    read_interleaved_segment_bytes,
    fromfile)
from nptdms.daqmx import DaqmxSegment
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


def read_segment_metadata(file, previous_segment_objects, previous_segment=None):
    (position, toc_mask, endianness, data_position, raw_data_offset,
     next_segment_offset, next_segment_pos) = read_lead_in(file)

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
        file, previous_segment_objects, previous_segment)
    return segment


def read_lead_in(file):
    position = file.tell()
    # First four bytes should be TDSm
    try:
        tag = file.read(4).decode('utf-8')
    except UnicodeDecodeError:
        raise ValueError("Segment does not start with TDSm")
    if tag == '':
        raise EOFError
    if tag != 'TDSm':
        raise ValueError(
            "Segment does not start with TDSm, but with %s" % tag)

    log.debug("Reading segment at %d", position)

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
    data_position = position + lead_size + raw_data_offset
    if next_segment_offset == 0xFFFFFFFFFFFFFFFF:
        # Segment size is unknown. This can happen if Labview crashes.
        # Try to read until the end of the file.
        log.warning(
            "Last segment of file has unknown size, "
            "will attempt to read to the end of the file")
        current_pos = file.tell()
        file.seek(0, os.SEEK_END)
        end_pos = file.tell()
        file.seek(current_pos, os.SEEK_SET)

        next_segment_pos = end_pos
        next_segment_offset = end_pos - position - lead_size
    else:
        log.debug("Next segment offset = %d, raw data offset = %d",
                  next_segment_offset, raw_data_offset)
        log.debug("Data size = %d b",
                  next_segment_offset - raw_data_offset)
        next_segment_pos = (
                position + next_segment_offset + lead_size)

    return (position, toc_mask, endianness, data_position, raw_data_offset,
            next_segment_offset, next_segment_pos)


class InterleavedDataSegment(BaseSegment):
    """ A TDMS segment with interleaved data
    """

    __slots__ = []

    def _new_segment_object(self, object_path):
        return TdmsSegmentObject(object_path, self.endianness)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        # If all data types have numpy types and all the lengths are
        # the same, then we can read all data at once with numpy,
        # which is much faster
        all_numpy = all(
            o.data_type.nptype is not None for o in data_objects)
        same_length = (len(
            set((o.number_values for o in data_objects))) == 1)
        if all_numpy and same_length:
            return self._read_interleaved_numpy(file, data_objects)
        else:
            return self._read_interleaved(file, data_objects)

    def _read_interleaved_numpy(self, file, data_objects):
        """Read interleaved data where all channels have a numpy type"""

        log.debug("Reading interleaved data all at once")

        # For non-DAQmx, simply use the data type sizes
        all_channel_bytes = sum(o.data_type.size for o in data_objects)
        log.debug("all_channel_bytes: %d", all_channel_bytes)

        # Read all data into 1 byte unsigned ints first
        combined_data = read_interleaved_segment_bytes(
            file, all_channel_bytes, data_objects[0].number_values)

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

    def _read_interleaved(self, file, data_objects):
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
                        obj.read_value(file))
                    points_added[obj.path] += 1

        return DataChunk.channel_data(object_data)


class ContiguousDataSegment(BaseSegment):
    """ A TDMS segment with contiguous (non-interleaved) data
    """

    __slots__ = []

    def _new_segment_object(self, object_path):
        return TdmsSegmentObject(object_path, self.endianness)

    def _read_data_chunk(self, file, data_objects, chunk_index):
        log.debug("Data is contiguous")
        object_data = {}
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            object_data[obj.path] = obj.read_values(file, number_values)
        return DataChunk.channel_data(object_data)

    def _read_channel_data_chunk(self, file, data_objects, chunk_index, channel_path):
        """ Read data from a chunk for a single channel
        """
        channel_data = ChannelDataChunk.empty()
        for obj in data_objects:
            number_values = self._get_channel_number_values(obj, chunk_index)
            if obj.path == channel_path:
                channel_data = ChannelDataChunk.channel_data(obj.read_values(file, number_values))
            elif number_values == obj.number_values:
                # Seek over data for other channel data
                file.seek(obj.data_size, os.SEEK_CUR)
            else:
                # In last chunk with reduced chunk size
                if obj.data_type.size is None:
                    # Type is unsized (eg. string), try reading number of values
                    obj.read_values(file, number_values)
                else:
                    file.seek(obj.data_type.size * number_values, os.SEEK_CUR)
        return channel_data

    def _get_channel_number_values(self, obj, chunk_index):
        if (chunk_index == (self.num_chunks - 1) and
                self.final_chunk_proportion != 1.0):
            return int(obj.number_values * self.final_chunk_proportion)
        else:
            return obj.number_values


class TdmsSegmentObject(BaseSegmentObject):
    """ A standard (non DAQmx) TDMS segment object
    """

    __slots__ = []

    def read_raw_data_index(self, f, raw_data_index_header):
        # Metadata format is standard (non-DAQmx) TDMS format.
        # raw_data_index_header gives the length of the index information.

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
        dimension = types.Uint32.read(f, self.endianness)
        # In TDMS version 2.0, 1 is the only valid value for dimension
        if dimension != 1:
            log.warning("Data dimension is not 1")

        # Read number of values
        self.number_values = types.Uint64.read(f, self.endianness)

        # Variable length data types have total size
        if self.data_type in (types.String,):
            self.data_size = types.Uint64.read(f, self.endianness)
        else:
            self.data_size = self.number_values * self.data_type.size

        log.debug(
            "Object number of values in segment: %d", self.number_values)

    @property
    def total_raw_data_width(self):
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
