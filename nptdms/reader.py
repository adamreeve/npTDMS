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
                    self._file, self.objects, previous_segment)

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


class DataSegment(object):
    """Data read from a single TDMS segment

    :ivar raw_data: A dictionary of object data in this segment for
        normal objects. Keys are object paths and values are numpy arrays.
    :ivar daqmx_raw_data: A dictionary of data in this segment for
        DAQmx raw data. Keys are object paths and values are dictionaries of
        numpy arrays keyed by scaler id.
    """

    def __init__(self, data, daqmx_data):
        self.raw_data = data
        self.daqmx_raw_data = daqmx_data
