""" Responsible for storing data read from TDMS files
"""

import tempfile
import numpy as np

from nptdms import types
from nptdms.log import log_manager

log = log_manager.get_logger(__name__)


def get_data_store(obj, memmap_dir=None):
    """Return a new data store to use for the given TDMS channel object

    :param obj: TDMS channel object to store data for
    :memmap_dir: Optional directory to store memmap files,
        or None to not use mmemap files
    """
    if obj.data_type is None:
        return None

    if obj.data_type == types.DaqMxRawData:
        return DaqmxDataStore(obj, memmap_dir)

    if obj.data_type.nptype is None:
        return ListDataStore()

    return NumpyDataStore(obj, memmap_dir)


class ListDataStore(object):
    """Simple list based data store for objects that don't have a
        corresponding numpy data type

       :ivar data: List of data points
    """

    def __init__(self):
        """Initialise new data store for a TDMS object
        """
        self.data = []

    def append_data(self, data):
        """Append data from a segment
        """
        self.data.extend(data)


class NumpyDataStore(object):
    """Stores data for a TDMS object in a numpy array

    :ivar data: Data that has been read for the object
    """

    def __init__(self, obj, memmap_dir=None):
        """Initialise data store backed by a numpy array

        :param obj: Object to store data for
        :param memmap_dir: Optional directory to store memmap files in.
        """

        self.path = obj.path
        self.data = _new_numpy_array(
            obj.data_type.nptype, obj.number_values, memmap_dir)
        self._data_insert_position = 0
        log.debug(
            "Allocated %d sample slots for %s", len(self.data), obj.path)

    def append_data(self, new_data):
        """Update the object data with a new array of data"""

        log.debug(
            "Adding %d data points to data for %s", len(new_data), self.path)
        start_pos = self._data_insert_position
        end_pos = self._data_insert_position + len(new_data)
        self.data[start_pos:end_pos] = new_data
        self._data_insert_position += len(new_data)


class DaqmxDataStore(object):
    """Stores raw scaler data for a DAQmx object in numpy arrays

    :ivar scaler_data: Dictionary mapping from scaler id to data for a scaler
    """

    def __init__(self, obj, memmap_dir=None):
        """Initialise data store backed by a numpy array

        :param obj: Object to store data for
        :param memmap_dir: Optional directory to store memmap files in.
        """

        self.path = obj.path
        self.scaler_data = {}
        self._scaler_insert_positions = {}
        for scaler in obj.scalers:
            self.scaler_data[scaler.scale_id] = _new_numpy_array(
                scaler.data_type.nptype, obj.number_values, memmap_dir)
            self._scaler_insert_positions[scaler.scale_id] = 0

    def append_data(self, scale_id, new_data):
        """Append new DAQmx scaler data read from a segment
        """

        log.debug("Adding %d data points for object %s, scaler %d",
                  len(new_data), self.path, scale_id)
        data_array = self.scaler_data[scale_id]
        start_pos = self._scaler_insert_positions[scale_id]
        end_pos = start_pos + len(new_data)
        data_array[start_pos:end_pos] = new_data
        self._scaler_insert_positions[scale_id] += len(new_data)


def _new_numpy_array(dtype, num_values, memmap_dir=None):
    """Initialise a new numpy array for data

    :param dtype: Numpy datatype for array
    :param num_values: Capacity required
    :param mmemap_dir: Optional directory to store memmap files
    """
    if memmap_dir:
        memmap_file = tempfile.NamedTemporaryFile(
            mode='w+b', prefix="nptdms_", dir=memmap_dir)
        return np.memmap(
            memmap_file.file,
            mode='w+',
            shape=(num_values,),
            dtype=dtype)

    return np.zeros(num_values, dtype=dtype)
