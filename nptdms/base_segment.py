from copy import copy
from io import UnsupportedOperation
import numpy as np

from nptdms import types
from nptdms.common import toc_properties
from nptdms.log import log_manager


log = log_manager.get_logger(__name__)


class BaseSegment(object):
    """ Abstract base class for TDMS segments
    """

    __slots__ = [
        'position', 'num_chunks', 'ordered_objects', 'toc_mask',
        'next_segment_offset', 'next_segment_pos',
        'raw_data_offset', 'data_position', 'final_chunk_proportion',
        'endianness']

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
            try:
                self.ordered_objects = previous_segment.ordered_objects
                self._calculate_chunks()
                return
            except AttributeError:
                raise ValueError(
                    "kTocMetaData is not set for segment but "
                    "there is no previous segment")

        if not self.toc_mask & toc_properties['kTocNewObjList']:
            # In this case, there can be a list of new objects that
            # are appended, or previous objects can also be repeated
            # if their properties change
            self.ordered_objects = [
                copy(o) for o in previous_segment.ordered_objects]

        log.debug("Reading segment object metadata at %d", file.tell())

        # First four bytes have number of objects in metadata
        num_objects = types.Int32.read(file, self.endianness)

        for _ in range(num_objects):
            # Read the object path
            object_path = types.String.read(file, self.endianness)

            # Add this segment object to the list of segment objects,
            # re-using any properties from previous segments.
            updating_existing = False
            segment_obj = None
            if not self.toc_mask & toc_properties['kTocNewObjList']:
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
                    segment_obj = self._new_segment_object(object_path)
                self.ordered_objects.append(segment_obj)
            # Read the metadata for this object, updating any
            # data structure information and properties.
            segment_obj.read_metadata(file)

        self._calculate_chunks()

    def read_raw_data(self, f):
        """Read raw data from a TDMS segment

        :returns: A generator of DataChunk objects with raw channel data for
            objects in this segment.
        """

        if not self.toc_mask & toc_properties['kTocRawData']:
            yield DataChunk.empty()

        f.seek(self.data_position)

        total_data_size = self.next_segment_offset - self.raw_data_offset
        log.debug(
            "Reading %d bytes of data at %d in %d chunks",
            total_data_size, f.tell(), self.num_chunks)

        data_objects = [o for o in self.ordered_objects if o.has_data]
        for chunk in range(self.num_chunks):
            yield self._read_data_chunk(f, data_objects, chunk)

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

    def _read_data_chunk(self, file, data_objects, chunk_index):
        raise NotImplementedError("Data chunk reading must be implemented in base classes")

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
        'has_data', 'data_type', 'endianness',
        'properties']

    def __init__(self, path, endianness):
        self.path = path
        self.endianness = endianness

        self.number_values = 0
        self.data_size = 0
        self.has_data = True
        self.data_type = None
        self.properties = None

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
        elif raw_data_index == 0x00000000:
            log.debug(
                "Object has same data structure as in the previous segment")
            self.has_data = True
        else:
            # New segment metadata, or updates to existing data
            self.has_data = True
            self.read_segment_metadata(f, raw_data_index)

        # Read data properties
        num_properties = types.Uint32.read(f, self.endianness)
        if num_properties > 0:
            log.debug("Reading %d properties", num_properties)
            self.properties = [
                read_property(f, self.endianness)
                for _ in range(num_properties)]

    def read_segment_metadata(self, file, raw_data_index):
        raise NotImplementedError("Segment metadata reading must be implemented in base classes")

    @property
    def total_raw_data_width(self):
        raise NotImplementedError("Raw data width must be implemented in base classes")

    @property
    def scaler_data_types(self):
        return None


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
