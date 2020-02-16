""" Lower level TDMS reader API that allows finer grained reading of data
"""

from nptdms.utils import Timer, OrderedDict
from nptdms.tdms_segment import TdmsSegment
from nptdms.log import log_manager

log = log_manager.get_logger(__name__)


class TdmsReader(object):
    """ Reads metadata and data from a TDMS file.

    :ivar objects: Dictionary of object path to TdmsObject
    """

    def __init__(self, tdms_file):
        """ Initialise a new TdmsReader

        :param file: An opened file object.
        """
        self._file = tdms_file
        self._segments = None
        self._prev_segment_objects = {}
        self.objects = OrderedDict()

    def read_metadata(self):
        """ Read all metadata and structure information from a TdmsFile
        """

        with Timer(log, "Read metadata"):
            # Read metadata first to work out how much space we need
            previous_segment = None
            while True:
                try:
                    segment = TdmsSegment(self._file)
                except EOFError:
                    # We've finished reading the file
                    break
                segment.read_metadata(
                    self._file, self._prev_segment_objects, previous_segment)

                self._update_object_state(segment.ordered_objects)
                self._segments.append(segment)
                previous_segment = segment

                if segment.next_segment_pos is None:
                    break
                else:
                    self._file.seek(segment.next_segment_pos)

    def read_data(self):
        """ A generator that returns data segments

        :returns: A generator that yields DataSegment objects
        """
        if self._segments is None:
            raise RuntimeError(
                "Cannot read data unless metadata has first been read")
        # TODO: read data from segments

    def _update_object_state(self, segment_objects):
        for segment_object in segment_objects:
            path = segment_object.path
            self._prev_segment_objects[path] = segment_object

            try:
                obj = self.objects[path]
            except KeyError:
                obj = ObjectState()
                self.objects[path] = obj
            for prop, val in segment_object.properties.items():
                obj[prop] = val
            if segment_object.has_data:
                obj.has_data = True
            obj.num_values += segment_object.num_values
            if (obj.data_type is not None and
                    obj.data_type != segment_object.data_type):
                raise ValueError(
                    "Segment data doesn't have the same type as previous "
                    "segments for objects %s. Expected type %s but got %s" %
                    (path, obj.data_type, segment_object.data_type))
            obj.data_type = segment_object.data_type


class ObjectState(object):
    def __init__(self):
        self.properties = {}
        self.data_type = None
        self.num_values = 0
        self.has_data = True
